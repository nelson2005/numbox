	.file	"<string>"
	.section	.ltext,"axl",@progbits
	.globl	_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd
	.p2align	4
	.type	_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd,@function
_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	vmovsd	%xmm0, 24(%rsp)
	movq	%rdi, %rbx
	movq	$0, (%rsp)
	testq	%rdx, %rdx
	jle	.LBB0_1
	movabsq	$numbox_pxy_cc_small_c65308d8847093c0, %r14
	leaq	8(%rsp), %r12
	movq	%rdx, %r15
	movq	%rsi, 16(%rsp)
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%ebp, %ebp
	movq	%rsp, %r13
	jmp	.LBB0_4
	.p2align	4
.LBB0_6:
	vmovsd	32(%rsp), %xmm0
	incq	%rbp
	vaddsd	8(%rsp), %xmm0, %xmm0
	cmpq	%rbp, %r15
	je	.LBB0_2
.LBB0_4:
	vmovsd	%xmm0, 32(%rsp)
	vcvtsi2sd	%rbp, %xmm1, %xmm0
	vaddsd	24(%rsp), %xmm0, %xmm0
	movq	$0, 8(%rsp)
	movq	%r12, %rdi
	movq	%r13, %rsi
	callq	*%r14
	testl	%eax, %eax
	je	.LBB0_6
	cmpl	$-2, %eax
	je	.LBB0_6
	movq	(%rsp), %rcx
	movq	16(%rsp), %rdx
	movq	%rcx, (%rdx)
	jmp	.LBB0_8
.LBB0_1:
	vxorpd	%xmm0, %xmm0, %xmm0
.LBB0_2:
	vmovsd	%xmm0, (%rbx)
	xorl	%eax, %eax
.LBB0_8:
	addq	$40, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd, .Lfunc_end0-_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd
	.cfi_endproc

	.globl	_ZN7cpython8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd
	.p2align	4
	.type	_ZN7cpython8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd,@function
_ZN7cpython8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$40, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rsi, %rdi
	movabsq	$.const.const_small_sum, %rsi
	movabsq	$PyArg_UnpackTuple, %r10
	leaq	32(%rsp), %r8
	leaq	24(%rsp), %r9
	movl	$2, %edx
	movl	$2, %ecx
	xorl	%eax, %eax
	callq	*%r10
	movq	$0, (%rsp)
	testl	%eax, %eax
	je	.LBB1_18
	movabsq	$_ZN08NumbaEnv8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd, %rax
	cmpq	$0, (%rax)
	je	.LBB1_21
	movq	32(%rsp), %rdi
	movabsq	$PyNumber_Long, %rax
	callq	*%rax
	movabsq	$Py_DecRef, %r15
	testq	%rax, %rax
	je	.LBB1_22
	movabsq	$PyLong_AsLongLong, %r12
	movq	%rax, %r14
	movq	%r14, %rdi
	callq	*%r12
	movq	%r14, %rdi
	movq	%rax, %rbx
	callq	*%r15
	movabsq	$PyErr_Occurred, %r12
	callq	*%r12
	testq	%rax, %rax
	jne	.LBB1_18
.LBB1_4:
	movq	24(%rsp), %rdi
	movabsq	$PyNumber_Float, %rax
	callq	*%rax
	movq	%rax, %r14
	movabsq	$PyFloat_AsDouble, %rax
	movq	%r14, %rdi
	callq	*%rax
	movq	%r14, %rdi
	vmovsd	%xmm0, 16(%rsp)
	callq	*%r15
	callq	*%r12
	testq	%rax, %rax
	jne	.LBB1_18
	vmovsd	16(%rsp), %xmm0
	movabsq	$_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd, %rax
	leaq	8(%rsp), %rdi
	movq	$0, 8(%rsp)
	movq	%rsp, %rsi
	movq	%rbx, %rdx
	callq	*%rax
	testl	%eax, %eax
	je	.LBB1_10
	jle	.LBB1_11
	movq	(%rsp), %rbx
	movabsq	$PyErr_Clear, %rax
	callq	*%rax
	movl	8(%rbx), %esi
	movq	(%rbx), %rdi
	cmpl	$0, 32(%rbx)
	jle	.LBB1_15
	movslq	%esi, %rsi
	movabsq	$PyBytes_FromStringAndSize, %rax
	callq	*%rax
	movq	16(%rbx), %rdi
	movq	%rax, %r14
	callq	*24(%rbx)
	testq	%rax, %rax
	je	.LBB1_23
	movabsq	$numba_runtime_build_excinfo_struct, %rcx
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	*%rcx
	movabsq	$NRT_Free, %r15
	movq	%rbx, %rdi
	movq	%rax, %r14
	callq	*%r15
	testq	%r14, %r14
	jne	.LBB1_16
	jmp	.LBB1_18
.LBB1_10:
	vmovsd	8(%rsp), %xmm0
	movabsq	$PyFloat_FromDouble, %rax
	callq	*%rax
	jmp	.LBB1_19
.LBB1_11:
	cmpl	$-3, %eax
	je	.LBB1_20
	cmpl	$-1, %eax
	je	.LBB1_18
	movabsq	$PyExc_SystemError, %rdi
	movabsq	$".const.unknown error when calling native function", %rsi
.LBB1_14:
	movabsq	$PyErr_SetString, %rax
	callq	*%rax
	jmp	.LBB1_18
.LBB1_15:
	movq	16(%rbx), %rdx
	movabsq	$numba_unpickle, %rax
	callq	*%rax
	movq	%rax, %r14
	testq	%r14, %r14
	je	.LBB1_18
.LBB1_16:
	movabsq	$numba_do_raise, %rax
	movq	%r14, %rdi
.LBB1_17:
	callq	*%rax
.LBB1_18:
	xorl	%eax, %eax
.LBB1_19:
	addq	$40, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB1_20:
	.cfi_def_cfa_offset 80
	movabsq	$PyExc_StopIteration, %rdi
	movabsq	$PyErr_SetNone, %rax
	jmp	.LBB1_17
.LBB1_21:
	movabsq	$PyExc_RuntimeError, %rdi
	movabsq	$".const.missing Environment: _ZN08NumbaEnv8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd", %rsi
	jmp	.LBB1_14
.LBB1_22:
	xorl	%ebx, %ebx
	movabsq	$PyErr_Occurred, %r12
	callq	*%r12
	testq	%rax, %rax
	je	.LBB1_4
	jmp	.LBB1_18
.LBB1_23:
	movabsq	$PyExc_RuntimeError, %rdi
	movabsq	$".const.Error creating Python tuple from runtime exception arguments", %rsi
	jmp	.LBB1_14
.Lfunc_end1:
	.size	_ZN7cpython8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd, .Lfunc_end1-_ZN7cpython8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd
	.cfi_endproc

	.globl	cfunc._ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd
	.p2align	4
	.type	cfunc._ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd,@function
cfunc._ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$32, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	%rdi, %rdx
	movabsq	$_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd, %rax
	leaq	16(%rsp), %rdi
	leaq	8(%rsp), %rsi
	movq	$0, 16(%rsp)
	movq	$0, 8(%rsp)
	callq	*%rax
	vmovsd	16(%rsp), %xmm0
	movq	8(%rsp), %rbx
	movl	$0, 4(%rsp)
	testl	%eax, %eax
	je	.LBB2_8
	movl	%eax, %ebp
	movabsq	$numba_gil_ensure, %rax
	leaq	4(%rsp), %rdi
	vmovsd	%xmm0, 24(%rsp)
	callq	*%rax
	testl	%ebp, %ebp
	jle	.LBB2_9
	movabsq	$PyErr_Clear, %rax
	callq	*%rax
	movl	8(%rbx), %esi
	movq	(%rbx), %rdi
	cmpl	$0, 32(%rbx)
	jle	.LBB2_12
	movslq	%esi, %rsi
	movabsq	$PyBytes_FromStringAndSize, %rax
	callq	*%rax
	movq	16(%rbx), %rdi
	movq	%rax, %r14
	callq	*24(%rbx)
	testq	%rax, %rax
	je	.LBB2_4
	movabsq	$numba_runtime_build_excinfo_struct, %rcx
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	*%rcx
	movabsq	$NRT_Free, %rbp
	movq	%rbx, %rdi
	movq	%rax, %r14
	callq	*%rbp
	testq	%r14, %r14
	jne	.LBB2_14
	jmp	.LBB2_7
.LBB2_9:
	cmpl	$-3, %ebp
	je	.LBB2_5
	cmpl	$-1, %ebp
	je	.LBB2_7
	movabsq	$PyExc_SystemError, %rdi
	movabsq	$".const.unknown error when calling native function.2", %rsi
	movabsq	$PyErr_SetString, %rax
	callq	*%rax
	jmp	.LBB2_7
.LBB2_12:
	movq	16(%rbx), %rdx
	movabsq	$numba_unpickle, %rax
	callq	*%rax
	movq	%rax, %r14
	testq	%r14, %r14
	je	.LBB2_7
.LBB2_14:
	movabsq	$numba_do_raise, %rax
	movq	%r14, %rdi
.LBB2_6:
	callq	*%rax
.LBB2_7:
	movabsq	$".const.<numba.core.cpu.CPUContext object at 0x747e4b002720>", %rdi
	movabsq	$PyUnicode_FromString, %rax
	callq	*%rax
	movabsq	$PyErr_WriteUnraisable, %r14
	movq	%rax, %rbx
	movq	%rbx, %rdi
	callq	*%r14
	movabsq	$Py_DecRef, %rax
	movq	%rbx, %rdi
	callq	*%rax
	movabsq	$numba_gil_release, %rax
	leaq	4(%rsp), %rdi
	callq	*%rax
	vmovsd	24(%rsp), %xmm0
.LBB2_8:
	addq	$32, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB2_5:
	.cfi_def_cfa_offset 64
	movabsq	$PyExc_StopIteration, %rdi
	movabsq	$PyErr_SetNone, %rax
	jmp	.LBB2_6
.LBB2_4:
	movabsq	$PyExc_RuntimeError, %rdi
	movabsq	$".const.Error creating Python tuple from runtime exception arguments.1", %rsi
	movabsq	$PyErr_SetString, %rax
	callq	*%rax
	vxorps	%xmm0, %xmm0, %xmm0
	jmp	.LBB2_8
.Lfunc_end2:
	.size	cfunc._ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd, .Lfunc_end2-cfunc._ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd
	.cfi_endproc

	.type	.const.const_small_sum,@object
	.section	.lrodata,"al",@progbits
.const.const_small_sum:
	.asciz	"const_small_sum"
	.size	.const.const_small_sum, 16

	.type	_ZN08NumbaEnv8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd,@object
	.comm	_ZN08NumbaEnv8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd,8,8
	.type	".const.missing Environment: _ZN08NumbaEnv8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd",@object
	.p2align	4, 0x0
".const.missing Environment: _ZN08NumbaEnv8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd":
	.asciz	"missing Environment: _ZN08NumbaEnv8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd"
	.size	".const.missing Environment: _ZN08NumbaEnv8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd", 109

	.type	".const.Error creating Python tuple from runtime exception arguments",@object
	.p2align	4, 0x0
".const.Error creating Python tuple from runtime exception arguments":
	.asciz	"Error creating Python tuple from runtime exception arguments"
	.size	".const.Error creating Python tuple from runtime exception arguments", 61

	.type	".const.unknown error when calling native function",@object
	.p2align	4, 0x0
".const.unknown error when calling native function":
	.asciz	"unknown error when calling native function"
	.size	".const.unknown error when calling native function", 43

	.type	".const.Error creating Python tuple from runtime exception arguments.1",@object
	.p2align	4, 0x0
".const.Error creating Python tuple from runtime exception arguments.1":
	.asciz	"Error creating Python tuple from runtime exception arguments"
	.size	".const.Error creating Python tuple from runtime exception arguments.1", 61

	.type	".const.unknown error when calling native function.2",@object
	.p2align	4, 0x0
".const.unknown error when calling native function.2":
	.asciz	"unknown error when calling native function"
	.size	".const.unknown error when calling native function.2", 43

	.type	".const.<numba.core.cpu.CPUContext object at 0x747e4b002720>",@object
	.p2align	4, 0x0
".const.<numba.core.cpu.CPUContext object at 0x747e4b002720>":
	.asciz	"<numba.core.cpu.CPUContext object at 0x747e4b002720>"
	.size	".const.<numba.core.cpu.CPUContext object at 0x747e4b002720>", 53

	.section	".note.GNU-stack","",@progbits
