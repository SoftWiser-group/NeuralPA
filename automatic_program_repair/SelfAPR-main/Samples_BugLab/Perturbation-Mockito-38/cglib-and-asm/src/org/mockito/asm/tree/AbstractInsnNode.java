[BugLab_Wrong_Literal]^public static final int INSN = -1;^48^^^^^43^53^public static final int INSN = 0;^[CLASS] AbstractInsnNode   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final int TYPE_INSN = 1;^63^^^^^58^68^public static final int TYPE_INSN = 3;^[CLASS] AbstractInsnNode   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final int FIELD_INSN = 3;^68^^^^^63^73^public static final int FIELD_INSN = 4;^[CLASS] AbstractInsnNode   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final int LINE = 13;^118^^^^^113^123^public static final int LINE = 14;^[CLASS] AbstractInsnNode   [VARIABLES] 
[BugLab_Variable_Misuse]^this.opcode = LOOKUPSWITCH_INSN;^149^^^^^148^151^this.opcode = opcode;^[CLASS] AbstractInsnNode  [METHOD] <init> [RETURN_TYPE] AbstractInsnNode(int)   final int opcode [VARIABLES] AbstractInsnNode  next  prev  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  index  opcode  boolean  
[BugLab_Wrong_Literal]^this.index = -TYPE_INSN;^150^^^^^148^151^this.index = -1;^[CLASS] AbstractInsnNode  [METHOD] <init> [RETURN_TYPE] AbstractInsnNode(int)   final int opcode [VARIABLES] AbstractInsnNode  next  prev  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  index  opcode  boolean  
[BugLab_Variable_Misuse]^return LOOKUPSWITCH_INSN;^159^^^^^158^160^return opcode;^[CLASS] AbstractInsnNode  [METHOD] getOpcode [RETURN_TYPE] int   [VARIABLES] AbstractInsnNode  next  prev  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  index  opcode  boolean  
[BugLab_Variable_Misuse]^return next;^178^^^^^177^179^return prev;^[CLASS] AbstractInsnNode  [METHOD] getPrevious [RETURN_TYPE] AbstractInsnNode   [VARIABLES] AbstractInsnNode  next  prev  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  index  opcode  boolean  
[BugLab_Variable_Misuse]^return prev;^189^^^^^188^190^return next;^[CLASS] AbstractInsnNode  [METHOD] getNext [RETURN_TYPE] AbstractInsnNode   [VARIABLES] AbstractInsnNode  next  prev  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  index  opcode  boolean  
[BugLab_Argument_Swapping]^return  ( LabelNode )  label.get ( map ) ;^216^^^^^215^217^return  ( LabelNode )  map.get ( label ) ;^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode   LabelNode label Map map [VARIABLES] LabelNode  label  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  index  opcode  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= clones.length; ++i )  {^228^^^^^226^232^for  ( int i = 0; i < clones.length; ++i )  {^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode[]   List labels Map map [VARIABLES] LabelNode[]  clones  List  labels  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  i  index  opcode  
[BugLab_Variable_Misuse]^clones[i] =  ( LabelNode )  map.get ( labels.get ( TYPE_INSN )  ) ;^229^^^^^226^232^clones[i] =  ( LabelNode )  map.get ( labels.get ( i )  ) ;^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode[]   List labels Map map [VARIABLES] LabelNode[]  clones  List  labels  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  i  index  opcode  
[BugLab_Argument_Swapping]^clones[i] =  ( LabelNode )  i.get ( labels.get ( map )  ) ;^229^^^^^226^232^clones[i] =  ( LabelNode )  map.get ( labels.get ( i )  ) ;^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode[]   List labels Map map [VARIABLES] LabelNode[]  clones  List  labels  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  i  index  opcode  
[BugLab_Argument_Swapping]^clones[i] =  ( LabelNode )  labels.get ( map.get ( i )  ) ;^229^^^^^226^232^clones[i] =  ( LabelNode )  map.get ( labels.get ( i )  ) ;^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode[]   List labels Map map [VARIABLES] LabelNode[]  clones  List  labels  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  i  index  opcode  
[BugLab_Variable_Misuse]^clones[i] =  ( LabelNode )  map.get ( labels.get ( VAR_INSN )  ) ;^229^^^^^226^232^clones[i] =  ( LabelNode )  map.get ( labels.get ( i )  ) ;^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode[]   List labels Map map [VARIABLES] LabelNode[]  clones  List  labels  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  i  index  opcode  
[BugLab_Argument_Swapping]^clones[i] =  ( LabelNode )  map.get ( i.get ( labels )  ) ;^229^^^^^226^232^clones[i] =  ( LabelNode )  map.get ( labels.get ( i )  ) ;^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode[]   List labels Map map [VARIABLES] LabelNode[]  clones  List  labels  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  i  index  opcode  
[BugLab_Wrong_Literal]^for  ( int i = MULTIANEWARRAY_INSN; i < clones.length; ++i )  {^228^^^^^226^232^for  ( int i = 0; i < clones.length; ++i )  {^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode[]   List labels Map map [VARIABLES] LabelNode[]  clones  List  labels  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  i  index  opcode  
[BugLab_Variable_Misuse]^clones[i] =  ( LabelNode )  map.get ( labels.get ( LOOKUPSWITCH_INSN )  ) ;^229^^^^^226^232^clones[i] =  ( LabelNode )  map.get ( labels.get ( i )  ) ;^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode[]   List labels Map map [VARIABLES] LabelNode[]  clones  List  labels  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  i  index  opcode  
[BugLab_Variable_Misuse]^clones[i] =  ( LabelNode )  map.get ( labels.get ( MULTIANEWARRAY_INSN )  ) ;^229^^^^^226^232^clones[i] =  ( LabelNode )  map.get ( labels.get ( i )  ) ;^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode[]   List labels Map map [VARIABLES] LabelNode[]  clones  List  labels  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  i  index  opcode  
