[BugLab_Wrong_Operator]^return new SourceValue ( type != null ? 1 : type.getSize (  )  ) ;^51^^^^^50^52^return new SourceValue ( type == null ? 1 : type.getSize (  )  ) ;^[CLASS] SourceInterpreter  [METHOD] newValue [RETURN_TYPE] Value   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Literal]^return new SourceValue ( type == null ? 0 : type.getSize (  )  ) ;^51^^^^^50^52^return new SourceValue ( type == null ? 1 : type.getSize (  )  ) ;^[CLASS] SourceInterpreter  [METHOD] newValue [RETURN_TYPE] Value   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Literal]^size = 3;^61^^^^^54^74^size = 2;^[CLASS] SourceInterpreter  [METHOD] newOperation [RETURN_TYPE] Value   AbstractInsnNode insn [VARIABLES] boolean  Object  cst  AbstractInsnNode  insn  int  size  
[BugLab_Wrong_Operator]^size = cst instanceof Long && cst instanceof Double ? 2 : 1;^65^^^^^54^74^size = cst instanceof Long || cst instanceof Double ? 2 : 1;^[CLASS] SourceInterpreter  [METHOD] newOperation [RETURN_TYPE] Value   AbstractInsnNode insn [VARIABLES] boolean  Object  cst  AbstractInsnNode  insn  int  size  
[BugLab_Wrong_Operator]^size = cst  <=  Long || cst instanceof Double ? 2 : 1;^65^^^^^54^74^size = cst instanceof Long || cst instanceof Double ? 2 : 1;^[CLASS] SourceInterpreter  [METHOD] newOperation [RETURN_TYPE] Value   AbstractInsnNode insn [VARIABLES] boolean  Object  cst  AbstractInsnNode  insn  int  size  
[BugLab_Wrong_Operator]^size = cst instanceof Long || cst  ||  Double ? 2 : 1;^65^^^^^54^74^size = cst instanceof Long || cst instanceof Double ? 2 : 1;^[CLASS] SourceInterpreter  [METHOD] newOperation [RETURN_TYPE] Value   AbstractInsnNode insn [VARIABLES] boolean  Object  cst  AbstractInsnNode  insn  int  size  
[BugLab_Wrong_Literal]^size = cst instanceof Long || cst instanceof Double ? size : 1;^65^^^^^54^74^size = cst instanceof Long || cst instanceof Double ? 2 : 1;^[CLASS] SourceInterpreter  [METHOD] newOperation [RETURN_TYPE] Value   AbstractInsnNode insn [VARIABLES] boolean  Object  cst  AbstractInsnNode  insn  int  size  
[BugLab_Wrong_Literal]^size = cst instanceof Long || cst instanceof Double ? 2 : 2;^65^^^^^54^74^size = cst instanceof Long || cst instanceof Double ? 2 : 1;^[CLASS] SourceInterpreter  [METHOD] newOperation [RETURN_TYPE] Value   AbstractInsnNode insn [VARIABLES] boolean  Object  cst  AbstractInsnNode  insn  int  size  
[BugLab_Wrong_Literal]^size = ;^71^^^^^54^74^size = 1;^[CLASS] SourceInterpreter  [METHOD] newOperation [RETURN_TYPE] Value   AbstractInsnNode insn [VARIABLES] boolean  Object  cst  AbstractInsnNode  insn  int  size  
[BugLab_Argument_Swapping]^return new SourceValue ( insn, size ) ;^73^^^^^54^74^return new SourceValue ( size, insn ) ;^[CLASS] SourceInterpreter  [METHOD] newOperation [RETURN_TYPE] Value   AbstractInsnNode insn [VARIABLES] boolean  Object  cst  AbstractInsnNode  insn  int  size  
[BugLab_Argument_Swapping]^return new SourceValue ( insn.getSize (  ) , value ) ;^77^^^^^76^78^return new SourceValue ( value.getSize (  ) , insn ) ;^[CLASS] SourceInterpreter  [METHOD] copyOperation [RETURN_TYPE] Value   AbstractInsnNode insn Value value [VARIABLES] boolean  Value  value  AbstractInsnNode  insn  
[BugLab_Wrong_Literal]^size = 1;^92^^^^^80^101^size = 2;^[CLASS] SourceInterpreter  [METHOD] unaryOperation [RETURN_TYPE] Value   AbstractInsnNode insn Value value [VARIABLES] boolean  Value  value  AbstractInsnNode  insn  int  size  
[BugLab_Wrong_Literal]^size = size;^98^^^^^80^101^size = 1;^[CLASS] SourceInterpreter  [METHOD] unaryOperation [RETURN_TYPE] Value   AbstractInsnNode insn Value value [VARIABLES] boolean  Value  value  AbstractInsnNode  insn  int  size  
[BugLab_Argument_Swapping]^return new SourceValue ( insn, size ) ;^100^^^^^80^101^return new SourceValue ( size, insn ) ;^[CLASS] SourceInterpreter  [METHOD] unaryOperation [RETURN_TYPE] Value   AbstractInsnNode insn Value value [VARIABLES] boolean  Value  value  AbstractInsnNode  insn  int  size  
[BugLab_Wrong_Literal]^size = ;^128^^^^^107^134^size = 2;^[CLASS] SourceInterpreter  [METHOD] binaryOperation [RETURN_TYPE] Value   AbstractInsnNode insn Value value1 Value value2 [VARIABLES] boolean  Value  value1  value2  AbstractInsnNode  insn  int  size  
[BugLab_Argument_Swapping]^return new SourceValue ( insn, size ) ;^133^^^^^107^134^return new SourceValue ( size, insn ) ;^[CLASS] SourceInterpreter  [METHOD] binaryOperation [RETURN_TYPE] Value   AbstractInsnNode insn Value value1 Value value2 [VARIABLES] boolean  Value  value1  value2  AbstractInsnNode  insn  int  size  
[BugLab_Wrong_Literal]^return new SourceValue ( 2, insn ) ;^142^^^^^136^143^return new SourceValue ( 1, insn ) ;^[CLASS] SourceInterpreter  [METHOD] ternaryOperation [RETURN_TYPE] Value   AbstractInsnNode insn Value value1 Value value2 Value value3 [VARIABLES] boolean  Value  value1  value2  value3  AbstractInsnNode  insn  
[BugLab_Wrong_Operator]^if  ( insn.getOpcode (  )  != MULTIANEWARRAY )  {^147^^^^^145^153^if  ( insn.getOpcode (  )  == MULTIANEWARRAY )  {^[CLASS] SourceInterpreter  [METHOD] naryOperation [RETURN_TYPE] Value   AbstractInsnNode insn List values [VARIABLES] boolean  List  values  AbstractInsnNode  insn  int  size  
[BugLab_Wrong_Literal]^size = 2;^148^^^^^145^153^size = 1;^[CLASS] SourceInterpreter  [METHOD] naryOperation [RETURN_TYPE] Value   AbstractInsnNode insn List values [VARIABLES] boolean  List  values  AbstractInsnNode  insn  int  size  
[BugLab_Wrong_Literal]^size = 0;^148^^^^^145^153^size = 1;^[CLASS] SourceInterpreter  [METHOD] naryOperation [RETURN_TYPE] Value   AbstractInsnNode insn List values [VARIABLES] boolean  List  values  AbstractInsnNode  insn  int  size  
[BugLab_Argument_Swapping]^return new SourceValue ( insn, size ) ;^152^^^^^145^153^return new SourceValue ( size, insn ) ;^[CLASS] SourceInterpreter  [METHOD] naryOperation [RETURN_TYPE] Value   AbstractInsnNode insn List values [VARIABLES] boolean  List  values  AbstractInsnNode  insn  int  size  
[BugLab_Wrong_Operator]^if  ( dv.insns instanceof SmallSet || dw.insns instanceof SmallSet )  {^158^^^^^155^173^if  ( dv.insns instanceof SmallSet && dw.insns instanceof SmallSet )  {^[CLASS] SourceInterpreter  [METHOD] merge [RETURN_TYPE] Value   Value v Value w [VARIABLES] SourceValue  dv  dw  Set  s  boolean  Value  v  w  
[BugLab_Wrong_Operator]^if  ( dv.insns  |  SmallSet && dw.insns instanceof SmallSet )  {^158^^^^^155^173^if  ( dv.insns instanceof SmallSet && dw.insns instanceof SmallSet )  {^[CLASS] SourceInterpreter  [METHOD] merge [RETURN_TYPE] Value   Value v Value w [VARIABLES] SourceValue  dv  dw  Set  s  boolean  Value  v  w  
[BugLab_Wrong_Operator]^if  ( dv.insns instanceof SmallSet && dw.insns  ^  SmallSet )  {^158^^^^^155^173^if  ( dv.insns instanceof SmallSet && dw.insns instanceof SmallSet )  {^[CLASS] SourceInterpreter  [METHOD] merge [RETURN_TYPE] Value   Value v Value w [VARIABLES] SourceValue  dv  dw  Set  s  boolean  Value  v  w  
[BugLab_Wrong_Operator]^if  ( s == dv.insns || dv.size == dw.size )  {^160^^^^^155^173^if  ( s == dv.insns && dv.size == dw.size )  {^[CLASS] SourceInterpreter  [METHOD] merge [RETURN_TYPE] Value   Value v Value w [VARIABLES] SourceValue  dv  dw  Set  s  boolean  Value  v  w  
[BugLab_Wrong_Operator]^if  ( s != dv.insns && dv.size == dw.size )  {^160^^^^^155^173^if  ( s == dv.insns && dv.size == dw.size )  {^[CLASS] SourceInterpreter  [METHOD] merge [RETURN_TYPE] Value   Value v Value w [VARIABLES] SourceValue  dv  dw  Set  s  boolean  Value  v  w  
[BugLab_Wrong_Operator]^if  ( s == dv.insns && dv.size < dw.size )  {^160^^^^^155^173^if  ( s == dv.insns && dv.size == dw.size )  {^[CLASS] SourceInterpreter  [METHOD] merge [RETURN_TYPE] Value   Value v Value w [VARIABLES] SourceValue  dv  dw  Set  s  boolean  Value  v  w  
[BugLab_Variable_Misuse]^return w;^161^^^^^155^173^return v;^[CLASS] SourceInterpreter  [METHOD] merge [RETURN_TYPE] Value   Value v Value w [VARIABLES] SourceValue  dv  dw  Set  s  boolean  Value  v  w  
[BugLab_Wrong_Operator]^if  ( s >= dv.insns && dv.size == dw.size )  {^160^^^^^155^173^if  ( s == dv.insns && dv.size == dw.size )  {^[CLASS] SourceInterpreter  [METHOD] merge [RETURN_TYPE] Value   Value v Value w [VARIABLES] SourceValue  dv  dw  Set  s  boolean  Value  v  w  
[BugLab_Wrong_Operator]^if  ( s == dv.insns && dv.size > dw.size )  {^160^^^^^155^173^if  ( s == dv.insns && dv.size == dw.size )  {^[CLASS] SourceInterpreter  [METHOD] merge [RETURN_TYPE] Value   Value v Value w [VARIABLES] SourceValue  dv  dw  Set  s  boolean  Value  v  w  
[BugLab_Wrong_Operator]^if  ( dv.size != dw.size && !dv.insns.containsAll ( dw.insns )  )  {^166^^^^^155^173^if  ( dv.size != dw.size || !dv.insns.containsAll ( dw.insns )  )  {^[CLASS] SourceInterpreter  [METHOD] merge [RETURN_TYPE] Value   Value v Value w [VARIABLES] SourceValue  dv  dw  Set  s  boolean  Value  v  w  
[BugLab_Wrong_Operator]^if  ( dv.size < dw.size || !dv.insns.containsAll ( dw.insns )  )  {^166^^^^^155^173^if  ( dv.size != dw.size || !dv.insns.containsAll ( dw.insns )  )  {^[CLASS] SourceInterpreter  [METHOD] merge [RETURN_TYPE] Value   Value v Value w [VARIABLES] SourceValue  dv  dw  Set  s  boolean  Value  v  w  
[BugLab_Variable_Misuse]^return w;^172^^^^^155^173^return v;^[CLASS] SourceInterpreter  [METHOD] merge [RETURN_TYPE] Value   Value v Value w [VARIABLES] SourceValue  dv  dw  Set  s  boolean  Value  v  w  
