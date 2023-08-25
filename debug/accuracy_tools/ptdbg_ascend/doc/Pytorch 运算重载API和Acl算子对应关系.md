工具Dump文件命名规则，`{api_type}_{api_name}_{api调用次数}_{正反向}_{输入输出}.index`, 如 Functional_conv2d_1_backward_input.0
Tensor___bool___0_forward_input.0
Torch_conv2d_1_backward_input.0

Pytorch运算符重载后API（主要涉及api_type为Tensor），工具当前支持Dump的运算符和ACL对应关系整理如下

| API  | ACL   | 运算符  |
| ------------ | ------------ | ------------ |
|  __	add	__ | Add  |  + |
| __	and	__  | BitwiseAnd  | &  |
| __	bool	__  | NonZero  | if x  |
| __	div	__  | RealDiv  |  / |
| __	ge	__  | GreaterEqual  | >=  |
| __	gt__  |  Greater |  > |
| __	iadd__  | Add  | +=  |
| __	iand__  | BitwiseAnd  | &=   |
| __	idiv__  | RealDiv  |  /= |
| __	ifloordiv__  | FloorDiv  |  //= |
| __	ilshift__  |  LeftShift | <<=  |
| __	imod__  |  FloorMod | %=  |
| __	imul__  |  Mul |  *= |
| __	ior__  |  BitwiseOr |   \|= |
| __	irshift__  |  RightShift |  >= |
| __	isub__  |  Sub | -=  |
| __	ixor__  | BitwiseXor  | ^=  |
| __	lshift__  |LeftShift   | <<  |
| __	matmul__  |  Dot |  @ |
| __	mod__  |  FloorMod |  % |
| __	mul__  | Mul  | *  |
| __	nonzero__  | NonZero  | 暂无  |
| __	or__  |  BitwiseOr | \|  |
| __	radd__  |  Add |  + |
| __	rmul__  |  Mul | *  |
| __	rshift__  | RightShift  | >>  |
| __	sub__  | Sub  |  - |
| __	truediv__  | RealDiv  |  / |
| __	xor__  |  BitwiseXor | ^  |
| floor_divide  |  FloorDiv |  // |
| __  getitem __ |  Strideslice |  [] |
