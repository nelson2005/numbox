	.file	"<string>"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI0_0:
	.quad	0x3ff0000000000000
	.section	.ltext,"axl",@progbits
	.globl	_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd
	.p2align	4
	.type	_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd,@function
_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd:
	testq	%rdx, %rdx
	jle	.LBB0_1
	movl	%edx, %eax
	andl	$7, %eax
	movabsq	$.LCPI0_0, %rsi
	cmpq	$8, %rdx
	jae	.LBB0_8
	vxorpd	%xmm1, %xmm1, %xmm1
	xorl	%ecx, %ecx
	jmp	.LBB0_4
.LBB0_1:
	vxorps	%xmm1, %xmm1, %xmm1
	xorl	%eax, %eax
	vmovsd	%xmm1, (%rdi)
	retq
.LBB0_8:
	vmovsd	(%rsi), %xmm2
	movabsq	$9223372036854775800, %rcx
	vxorpd	%xmm1, %xmm1, %xmm1
	andq	%rcx, %rdx
	xorl	%ecx, %ecx
	.p2align	4
.LBB0_9:
	vcvtsi2sd	%rcx, %xmm5, %xmm3
	leaq	1(%rcx), %r8
	vaddsd	%xmm3, %xmm0, %xmm3
	vaddsd	%xmm3, %xmm3, %xmm3
	vaddsd	%xmm2, %xmm3, %xmm3
	vaddsd	%xmm3, %xmm1, %xmm1
	vcvtsi2sd	%r8, %xmm5, %xmm3
	leaq	2(%rcx), %r8
	vcvtsi2sd	%r8, %xmm5, %xmm4
	leaq	3(%rcx), %r8
	vaddsd	%xmm3, %xmm0, %xmm3
	vaddsd	%xmm3, %xmm3, %xmm3
	vaddsd	%xmm2, %xmm3, %xmm3
	vaddsd	%xmm3, %xmm1, %xmm1
	vaddsd	%xmm4, %xmm0, %xmm3
	vaddsd	%xmm3, %xmm3, %xmm3
	vaddsd	%xmm2, %xmm3, %xmm3
	vaddsd	%xmm3, %xmm1, %xmm1
	vcvtsi2sd	%r8, %xmm5, %xmm3
	leaq	4(%rcx), %r8
	vaddsd	%xmm3, %xmm0, %xmm3
	vaddsd	%xmm3, %xmm3, %xmm3
	vaddsd	%xmm2, %xmm3, %xmm3
	vaddsd	%xmm3, %xmm1, %xmm1
	vcvtsi2sd	%r8, %xmm5, %xmm3
	leaq	5(%rcx), %r8
	vaddsd	%xmm3, %xmm0, %xmm3
	vaddsd	%xmm3, %xmm3, %xmm3
	vaddsd	%xmm2, %xmm3, %xmm3
	vaddsd	%xmm3, %xmm1, %xmm1
	vcvtsi2sd	%r8, %xmm5, %xmm3
	leaq	6(%rcx), %r8
	vaddsd	%xmm3, %xmm0, %xmm3
	vaddsd	%xmm3, %xmm3, %xmm3
	vaddsd	%xmm2, %xmm3, %xmm3
	vaddsd	%xmm3, %xmm1, %xmm1
	vcvtsi2sd	%r8, %xmm5, %xmm3
	leaq	7(%rcx), %r8
	addq	$8, %rcx
	vaddsd	%xmm3, %xmm0, %xmm3
	vaddsd	%xmm3, %xmm3, %xmm3
	vaddsd	%xmm2, %xmm3, %xmm3
	vaddsd	%xmm3, %xmm1, %xmm1
	vcvtsi2sd	%r8, %xmm5, %xmm3
	vaddsd	%xmm3, %xmm0, %xmm3
	vaddsd	%xmm3, %xmm3, %xmm3
	vaddsd	%xmm2, %xmm3, %xmm3
	vaddsd	%xmm3, %xmm1, %xmm1
	cmpq	%rdx, %rcx
	jne	.LBB0_9
.LBB0_4:
	testq	%rax, %rax
	je	.LBB0_7
	vmovsd	(%rsi), %xmm2
	.p2align	4
.LBB0_6:
	vcvtsi2sd	%rcx, %xmm5, %xmm3
	incq	%rcx
	decq	%rax
	vaddsd	%xmm3, %xmm0, %xmm3
	vaddsd	%xmm3, %xmm3, %xmm3
	vaddsd	%xmm2, %xmm3, %xmm3
	vaddsd	%xmm3, %xmm1, %xmm1
	jne	.LBB0_6
.LBB0_7:
	xorl	%eax, %eax
	vmovsd	%xmm1, (%rdi)
	retq
.Lfunc_end0:
	.size	_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd, .Lfunc_end0-_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd

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
	testl	%eax, %eax
	je	.LBB1_1
	movabsq	$_ZN08NumbaEnv8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd, %rax
	cmpq	$0, (%rax)
	je	.LBB1_4
	movq	32(%rsp), %rdi
	movabsq	$PyNumber_Long, %rax
	callq	*%rax
	movabsq	$Py_DecRef, %r15
	testq	%rax, %rax
	je	.LBB1_7
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
	jne	.LBB1_1
.LBB1_10:
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
	jne	.LBB1_1
	vmovsd	16(%rsp), %xmm0
	movabsq	$_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd, %rax
	leaq	8(%rsp), %rdi
	movq	%rbx, %rdx
	movq	$0, 8(%rsp)
	callq	*%rax
	testl	%eax, %eax
	je	.LBB1_17
	jg	.LBB1_18
	cmpl	$-1, %eax
	je	.LBB1_1
	cmpl	$-3, %eax
	jne	.LBB1_16
	movabsq	$PyExc_StopIteration, %rdi
	movabsq	$PyErr_SetNone, %rax
	callq	*%rax
	jmp	.LBB1_1
.LBB1_17:
	vmovsd	8(%rsp), %xmm0
	movabsq	$PyFloat_FromDouble, %rax
	callq	*%rax
	jmp	.LBB1_2
.LBB1_16:
	movabsq	$PyExc_SystemError, %rdi
	movabsq	$".const.unknown error when calling native function", %rsi
.LBB1_5:
	movabsq	$PyErr_SetString, %rax
	callq	*%rax
.LBB1_1:
	xorl	%eax, %eax
.LBB1_2:
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
.LBB1_4:
	.cfi_def_cfa_offset 80
	movabsq	$PyExc_RuntimeError, %rdi
	movabsq	$".const.missing Environment: _ZN08NumbaEnv8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd", %rsi
	jmp	.LBB1_5
.LBB1_7:
	xorl	%ebx, %ebx
	movabsq	$PyErr_Occurred, %r12
	callq	*%r12
	testq	%rax, %rax
	je	.LBB1_10
	jmp	.LBB1_1
.LBB1_18:
	movabsq	$PyErr_Clear, %rax
	callq	*%rax
.Lfunc_end1:
	.size	_ZN7cpython8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd, .Lfunc_end1-_ZN7cpython8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd
	.cfi_endproc

	.globl	cfunc._ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd
	.p2align	4
	.type	cfunc._ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd,@function
cfunc._ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	%rdi, %rdx
	movabsq	$_ZN8__main__15const_small_sumB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dExd, %rax
	leaq	16(%rsp), %rdi
	movq	$0, 16(%rsp)
	callq	*%rax
	vmovsd	16(%rsp), %xmm0
	movl	$0, 12(%rsp)
	testl	%eax, %eax
	je	.LBB2_5
	movl	%eax, %ebx
	movabsq	$numba_gil_ensure, %rax
	leaq	12(%rsp), %rdi
	vmovsd	%xmm0, 24(%rsp)
	callq	*%rax
	testl	%ebx, %ebx
	jg	.LBB2_2
	cmpl	$-1, %ebx
	je	.LBB2_4
	cmpl	$-3, %ebx
	jne	.LBB2_3
	movabsq	$PyExc_StopIteration, %rdi
	movabsq	$PyErr_SetNone, %rax
	callq	*%rax
	jmp	.LBB2_4
.LBB2_3:
	movabsq	$PyExc_SystemError, %rdi
	movabsq	$".const.unknown error when calling native function.2", %rsi
	movabsq	$PyErr_SetString, %rax
	callq	*%rax
.LBB2_4:
	movabsq	$".const.<numba.core.cpu.CPUContext object at 0x7252fda5cd10>", %rdi
	movabsq	$PyUnicode_FromString, %rax
	callq	*%rax
	movq	%rax, %rbx
	movabsq	$PyErr_WriteUnraisable, %rax
	movq	%rbx, %rdi
	callq	*%rax
	movabsq	$Py_DecRef, %rax
	movq	%rbx, %rdi
	callq	*%rax
	movabsq	$numba_gil_release, %rax
	leaq	12(%rsp), %rdi
	callq	*%rax
	vmovsd	24(%rsp), %xmm0
.LBB2_5:
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB2_2:
	.cfi_def_cfa_offset 48
	movabsq	$PyErr_Clear, %rax
	callq	*%rax
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

	.type	".const.unknown error when calling native function",@object
	.p2align	4, 0x0
".const.unknown error when calling native function":
	.asciz	"unknown error when calling native function"
	.size	".const.unknown error when calling native function", 43

	.type	".const.unknown error when calling native function.2",@object
	.p2align	4, 0x0
".const.unknown error when calling native function.2":
	.asciz	"unknown error when calling native function"
	.size	".const.unknown error when calling native function.2", 43

	.type	".const.<numba.core.cpu.CPUContext object at 0x7252fda5cd10>",@object
	.p2align	4, 0x0
".const.<numba.core.cpu.CPUContext object at 0x7252fda5cd10>":
	.asciz	"<numba.core.cpu.CPUContext object at 0x7252fda5cd10>"
	.size	".const.<numba.core.cpu.CPUContext object at 0x7252fda5cd10>", 53

	.type	_ZN08NumbaEnv13binding_small5smallB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEd,@object
	.comm	_ZN08NumbaEnv13binding_small5smallB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEd,8,8
	.section	".note.GNU-stack","",@progbits
