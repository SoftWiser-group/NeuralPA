[BugLab_Variable_Misuse]^super ( operand ) ;^56^^^^^55^58^super ( opcode ) ;^[CLASS] IntInsnNode  [METHOD] <init> [RETURN_TYPE] IntInsnNode(int,int)   final int opcode final int operand [VARIABLES] int  opcode  operand  boolean  
[BugLab_Variable_Misuse]^this.operand = opcode;^57^^^^^55^58^this.operand = operand;^[CLASS] IntInsnNode  [METHOD] <init> [RETURN_TYPE] IntInsnNode(int,int)   final int opcode final int operand [VARIABLES] int  opcode  operand  boolean  
[BugLab_Variable_Misuse]^this.opcode = operand;^67^^^^^66^68^this.opcode = opcode;^[CLASS] IntInsnNode  [METHOD] setOpcode [RETURN_TYPE] void   final int opcode [VARIABLES] int  opcode  operand  boolean  
[BugLab_Argument_Swapping]^mv.visitIntInsn ( operand, opcode ) ;^75^^^^^74^76^mv.visitIntInsn ( opcode, operand ) ;^[CLASS] IntInsnNode  [METHOD] accept [RETURN_TYPE] void   MethodVisitor mv [VARIABLES] int  opcode  operand  MethodVisitor  mv  boolean  
[BugLab_Argument_Swapping]^return new IntInsnNode ( operand, opcode ) ;^79^^^^^78^80^return new IntInsnNode ( opcode, operand ) ;^[CLASS] IntInsnNode  [METHOD] clone [RETURN_TYPE] AbstractInsnNode   Map labels [VARIABLES] Map  labels  int  opcode  operand  boolean  