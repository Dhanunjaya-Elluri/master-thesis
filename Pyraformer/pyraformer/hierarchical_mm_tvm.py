#!/usr/bin/env python
# coding: utf-8

"""Hierarchical TVM for Pyraformer model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"


import os.path
import sys

import torch

sys.path.append("pyraformer/tvm/python")


class GraphMM(torch.autograd.Function):
    """
    Custom PyTorch autograd function for graph-based matrix multiplication using TVM (Tensor Virtual Machine).

    This class encapsulates the TVM code for compiling and executing a specialized matrix multiplication function,
    which is optimized for certain types of operations common in neural networks, such as in attention mechanisms.
    It supports different data types and devices (CPU/GPU), and it allows for a high degree of customization
    in terms of tensor shapes and operations.

    Attributes:
        function_dict (dict): A dictionary to cache compiled TVM functions for different configurations.
    """

    function_dict = (
        {}
    )  # save a list of functions, each has a different set of parameters

    @staticmethod
    def _compile_function(
        dtype: str, device: str, b0: int = 4, b1: int = 8, b2: int = 8
    ):
        """
        Compiles a TVM function that performs a specialized form of matrix multiplication.

        The function is compiled for specific data types and devices, with configurable tensor tile sizes,
        which are critical for performance.

        Args:
            dtype (str): Data type of the tensors, e.g., 'float32'.
            device (str): Target device, 'cpu' or 'cuda'.
            b0, b1, b2 (int): Tile sizes for the tensor dimensions, important for performance tuning.

        Returns:
            The compiled TVM function.
        """
        import tvm  # import the full tvm library here for compilation.

        # Don't import at the top of the file in case we don't need to compile
        from tvm.contrib import nvcc

        @tvm.register_func
        def tvm_callback_cuda_compile(code):
            """Use nvcc compiler for better perf."""
            ptx = nvcc.compile_cuda(
                code, target="ptx", arch="sm_52"
            )  # use old arch for this to work on old GPUs
            return ptx

        assert dtype in ["float16", "float32", "float64"]
        assert device in ["cpu", "cuda"]
        device = None if device == "cpu" else device
        tgt_host = "llvm"

        b = tvm.te.var("b")  # batch size
        n = tvm.te.var("n")  # sequence length
        h = tvm.te.var("h")  # number of heads
        m = tvm.te.var("m")  # hidden dimension
        w = tvm.te.var("w")  # window size
        padding = tvm.te.var("padding")  # padding
        transpose_t1 = tvm.te.var("transpose_t1")  # t1 should be transposed
        t1d3 = tvm.te.var("t1d3")  # last dimension of t1
        t3d3 = tvm.te.var("t3d3")  # last dimension of t3 (the result tensor)
        max_attn = tvm.te.var("max_attn")
        X = tvm.te.placeholder((b, n, h, t1d3), name="X", dtype=dtype)  # first tensor
        Y = tvm.te.placeholder((b, n, h, m), name="Y", dtype=dtype)  # second tensor
        k = tvm.te.reduce_axis((0, t1d3), name="k")  # dimension to sum over
        q_k_mask = tvm.te.placeholder(
            (n, max_attn), name="q_k", dtype="int"
        )  # dilation per head
        k_q_mask = tvm.te.placeholder((n, max_attn), name="k_q", dtype="int")  #
        output_shape = (b, n, h, t3d3)  # shape of the result tensor

        algorithm = lambda l, i, q, j: tvm.te.sum(
            tvm.te.if_then_else(
                t3d3
                == m,  # if output dimension == m, then t1 is diagonaled (FIXME: This breaks if t3d3 == m == t1d3)
                tvm.te.if_then_else(
                    transpose_t1 == 0,
                    tvm.te.if_then_else(
                        q_k_mask[i, k] >= 0,
                        X[l, i, q, k] * Y[l, q_k_mask[i, k], q, j],  # t1 is diagonaled
                        padding,
                    ),
                    tvm.te.if_then_else(
                        q_k_mask[i, k] >= 0,
                        X[l, q_k_mask[i, k], q, k_q_mask[i, k]]
                        * Y[l, q_k_mask[i, k], q, j],
                        # # t1 is diagonaled and should be transposed
                        padding,
                    ),
                ),
                tvm.te.if_then_else(
                    q_k_mask[i, j] >= 0,
                    X[l, i, q, k] * Y[l, q_k_mask[i, j], q, k],
                    # t1 is not diagonaled, but the output tensor is going to be
                    padding,
                ),
            ),
            axis=k,
        )

        Z = tvm.te.compute(
            output_shape, algorithm, name="Z"
        )  # automatically generate cuda code
        s = tvm.te.create_schedule(Z.op)

        print(
            "Lowering: \n ===================== \n{}".format(
                tvm.lower(s, [X, Y, q_k_mask, k_q_mask], simple_mode=True)
            )
        )

        # split long axis into smaller chunks and assing each one to a separate GPU thread/block
        ko, ki = s[Z].split(Z.op.reduce_axis[0], factor=b0)
        ZF = s.rfactor(Z, ki)

        j_outer, j_inner = s[Z].split(s[Z].op.axis[-1], factor=b1)
        i_outer, i_inner = s[Z].split(s[Z].op.axis[1], factor=b2)

        s[Z].bind(j_outer, tvm.te.thread_axis("blockIdx.x"))
        s[Z].bind(j_inner, tvm.te.thread_axis("threadIdx.y"))

        s[Z].bind(i_outer, tvm.te.thread_axis("blockIdx.y"))
        s[Z].bind(i_inner, tvm.te.thread_axis("threadIdx.z"))

        tx = tvm.te.thread_axis("threadIdx.x")
        s[Z].bind(s[Z].op.reduce_axis[0], tx)
        s[ZF].compute_at(s[Z], s[Z].op.reduce_axis[0])
        s[Z].set_store_predicate(tx.var.equal(0))

        print(
            "Lowering with GPU splits: \n ===================== \n{}".format(
                tvm.lower(s, [X, Y, q_k_mask, k_q_mask], simple_mode=True)
            )
        )

        # compiling the automatically generated cuda code
        graph_mm = tvm.build(
            s,
            [X, Y, Z, q_k_mask, k_q_mask, max_attn, padding, transpose_t1, t3d3],
            target=device,
            target_host=tgt_host,
            name="graph_mm",
        )
        return graph_mm

    @staticmethod
    def _get_lib_filename(dtype: str, device: str) -> str:
        """
        Constructs the filename for the compiled TVM library based on data type and device.

        Args:
            dtype (str): Data type of the tensors.
            device (str): Target device.

        Returns:
            str: Filename for the compiled TVM library.
        """
        base_filename = "lib/lib_hierarchical_mm"
        return "{}_{}_{}.so".format(base_filename, dtype, device)

    @staticmethod
    def _save_compiled_function(f, dtype: str, device: str):
        """
        Saves the compiled TVM function to a file.

        Args:
            f: The compiled TVM function.
            dtype (str): Data type of the tensors.
            device (str): Target device.
        """
        if not os.path.exists("lib/"):
            os.makedirs("lib/")
        f.export_library(GraphMM._get_lib_filename(dtype, device))

    @staticmethod
    def _load_compiled_function(dtype: str, device: str):
        """
        Loads a compiled TVM function from a file.

        Args:
            dtype (str): Data type of the tensors.
            device (str): Target device.

        Returns:
            The loaded TVM function, if available.
        """
        # from tvm.module import load  # this can be the small runtime python library, and doesn't need to be the whole thing
        from tvm.runtime.module import load_module as load

        filename = GraphMM._get_lib_filename(dtype, device)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        potential_dirs = [
            "../../",
            "../",
            "./",
            f"{current_dir}/",
            f"{current_dir}/../",
        ]
        for potential_dir in potential_dirs:
            filepath = "{}{}".format(potential_dir, filename)
            if os.path.isfile(filepath):
                print("Loading tvm binary from: {}".format(filepath))
                return load(filepath)
        return None

    @staticmethod
    def _get_function(dtype: str, device: str):
        """
        Retrieves a compiled TVM function, either from cache, disk, or by compiling it.

        Args:
            dtype (str): Data type of the tensors.
            device (str): Target device.

        Returns:
            The TVM function for graph-based matrix multiplication.
        """
        # A list of arguments that define the function
        args = (dtype, device)
        if args not in GraphMM.function_dict:
            graph_mm = GraphMM._load_compiled_function(
                dtype, device
            )  # try to load from disk
            if not graph_mm:
                print("Tvm binary not found. Compiling ...")
                graph_mm = GraphMM._compile_function(dtype, device)  # compile
                GraphMM._save_compiled_function(graph_mm, dtype, device)  # save to disk
            # convert the tvm function into a pytorch function
            from tvm.contrib import dlpack

            graph_mm_pytorch = dlpack.to_pytorch_func(
                graph_mm
            )  # wrap it as a pytorch function
            # save the function into a dictionary to be reused
            GraphMM.function_dict[
                args
            ] = graph_mm_pytorch  # save it in a dictionary for next time
        return GraphMM.function_dict[args]

    @staticmethod
    def _graph_mm(
        t1: torch.Tensor,
        t2: torch.Tensor,
        q_k_mask: torch.Tensor,
        k_q_mask: torch.Tensor,
        is_t1_diagonaled: bool = False,
        transpose_t1: bool = False,
        padding: int = 0,
        autoregressive: bool = False,
    ) -> torch.Tensor:
        """
        Performs the graph-based matrix multiplication using the compiled TVM function.

        Args:
            t1 (torch.Tensor): First input tensor.
            t2 (torch.Tensor): Second input tensor.
            q_k_mask (torch.Tensor): Query-key mask tensor.
            k_q_mask (torch.Tensor): Key-query mask tensor.
            is_t1_diagonaled (bool): Indicates if t1 is diagonaled.
            transpose_t1 (bool): Indicates if t1 should be transposed.
            padding (int): Padding value for invalid locations.
            autoregressive (bool): Indicates if the operation is autoregressive.

        Returns:
            torch.Tensor: The result of the graph-based matrix multiplication.
        """
        dtype = str(t1.dtype).split(".")[1]
        device = t1.device.type
        assert len(t1.shape) == 4
        assert len(t1.shape) == len(t2.shape)
        assert t1.shape[:3] == t2.shape[:3]

        b = t1.shape[0]  # batch size
        n = t1.shape[1]  # sequence length
        h = t1.shape[2]  # number of heads
        m = t2.shape[3]  # hidden dimension
        max_attn = q_k_mask.size(1)
        if is_t1_diagonaled:
            assert t1.shape[3] == max_attn
            r = t1.new_empty(b, n, h, m)  # allocate spase for the result tensor
        else:
            assert not transpose_t1
            assert t1.shape[3] == m
            r = t1.new_empty(b, n, h, max_attn)  # allocate spase for the result tensor

        # gets function from memory, from disk or compiles it from scratch
        _graph_mm_function = GraphMM._get_function(dtype=dtype, device=device)

        # The last argument to this function is a little hacky. It is the size of the last dimension of the result tensor
        # We use it as a proxy to tell if t1_is_diagonaled or not (if t1 is diagonaled, result is not, and vice versa).
        # The second reason is that the lambda expression in `_compile_function` is easier to express when the shape
        # of the output is known
        # This functions computes diagonal_mm then saves the result in `r`
        if m == max_attn:
            # FIXME
            print(
                "Error: the hidden dimension {m} shouldn't match number of diagonals {c}"
            )
            assert False
        _graph_mm_function(
            t1,
            t2,
            r,
            q_k_mask,
            k_q_mask,
            max_attn,
            padding,
            transpose_t1,
            m if is_t1_diagonaled else max_attn,
        )
        return r

    @staticmethod
    def _prepare_tensors(t: torch.Tensor) -> torch.Tensor:
        """
        Prepares and fixes the stride information of the input tensor for TVM compatibility.

        Args:
            t (torch.Tensor): The tensor to prepare.

        Returns:
            torch.Tensor: The prepared tensor with fixed stride information.
        """
        assert t.is_contiguous()
        t_stride = list(t.stride())
        t_size = list(t.size())
        # Fix wrong stride information for the first dimension. This occures when batch_size=1
        if t_size[0] == 1 and t_stride[0] == t_stride[1]:
            # In this case, the stride of the first dimension should be the product
            # of the sizes  of all other dimensions
            t_stride[0] = t_size[1] * t_size[2] * t_size[3]
            t = t.as_strided(size=t_size, stride=t_stride)
        return t

    min_seq_len = 16  # Minimum sequence length to avoid splitting errors

    @staticmethod
    def forward(
        ctx,
        t1: torch.Tensor,
        t2: torch.Tensor,
        q_k_mask: torch.Tensor,
        k_q_mask: torch.Tensor,
        is_t1_diagonaled: bool = False,
        padding: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for the graph-based matrix multiplication.

        Computes the diagonal matrix multiplication of t1 and t2 using a compiled TVM function.

        Args:
            t1 (torch.Tensor): First input tensor.
            t2 (torch.Tensor): Second input tensor.
            q_k_mask (torch.Tensor): Query-key mask tensor.
            k_q_mask (torch.Tensor): Key-query mask tensor.
            is_t1_diagonaled (bool): Indicates if t1 is diagonaled.
            padding (int): Padding value for invalid locations.

        Returns:
            torch.Tensor: The result of the matrix multiplication.
        """
        seq_len = t1.size(1)
        assert (
            seq_len >= GraphMM.min_seq_len
        ), "avoid splitting errors by using seq_len >= {}".format(
            GraphMM.min_seq_len
        )  # FIXME

        t1 = GraphMM._prepare_tensors(t1)
        t2 = GraphMM._prepare_tensors(t2)
        q_k_mask = GraphMM._prepare_tensors(q_k_mask)
        k_q_mask = GraphMM._prepare_tensors(k_q_mask)
        ctx.save_for_backward(t1, t2, q_k_mask, k_q_mask)
        ctx.is_t1_diagonaled = is_t1_diagonaled
        # output = t1.mm(t2)  # what would have been called if this was a regular matmul
        output = GraphMM._graph_mm(
            t1,
            t2,
            q_k_mask,
            k_q_mask,
            is_t1_diagonaled=is_t1_diagonaled,
            padding=padding,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass for the graph-based matrix multiplication.

        Computes the gradients for t1 and t2 based on the gradient of the output.

        Args:
            grad_output (torch.Tensor): Gradient of the output tensor.

        Returns:
            tuple: Gradients for t1, t2, and None for the remaining arguments.
        """
        t1, t2, q_k_mask, k_q_mask = ctx.saved_tensors
        is_t1_diagonaled = ctx.is_t1_diagonaled
        if not grad_output.is_contiguous():
            grad_output = (
                grad_output.contiguous()
            )  # tvm requires all input tensors to be contiguous
        grad_output = GraphMM._prepare_tensors(grad_output)
        # http://cs231n.github.io/optimization-2/
        # https://pytorch.org/docs/master/notes/extending.html
        # grad_t1 = grad_output.mm(t2)  # what would have been called if this was a regular matmul
        grad_t1 = GraphMM._graph_mm(
            grad_output, t2, q_k_mask, k_q_mask, is_t1_diagonaled=not is_t1_diagonaled
        )
        # grad_t2 = grad_output.t().mm(t1)  # or `grad_t2 = t1.t().mm(grad_output).t()` because `(AB)^T = B^TA^T`
        if is_t1_diagonaled:
            grad_t2 = GraphMM._graph_mm(
                t1,
                grad_output,
                q_k_mask,
                k_q_mask,
                is_t1_diagonaled=True,
                transpose_t1=True,
            )
        else:
            grad_t2 = GraphMM._graph_mm(
                grad_output,
                t1,
                q_k_mask,
                k_q_mask,
                is_t1_diagonaled=True,
                transpose_t1=True,
            )
        return grad_t1, grad_t2, None, None, None, None, None


graph_mm = GraphMM.apply
