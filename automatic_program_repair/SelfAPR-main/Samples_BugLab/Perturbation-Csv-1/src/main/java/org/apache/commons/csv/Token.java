[BugLab_Wrong_Literal]^private static final int INITIAL_TOKEN_LENGTH = 4;^31^^^^^26^36^private static final int INITIAL_TOKEN_LENGTH = 50;^[CLASS] Token Type   [VARIABLES] 
[BugLab_Wrong_Literal]^content.setLength ( INITIAL_TOKEN_LENGTH ) ;^57^^^^^56^61^content.setLength ( 0 ) ;^[CLASS] Token Type  [METHOD] reset [RETURN_TYPE] Token   [VARIABLES] Type  EOF  EORECORD  INVALID  TOKEN  type  boolean  isReady  StringBuilder  content  int  INITIAL_TOKEN_LENGTH  
[BugLab_Variable_Misuse]^type = TOKEN;^58^^^^^56^61^type = INVALID;^[CLASS] Token Type  [METHOD] reset [RETURN_TYPE] Token   [VARIABLES] Type  EOF  EORECORD  INVALID  TOKEN  type  boolean  isReady  StringBuilder  content  int  INITIAL_TOKEN_LENGTH  
[BugLab_Wrong_Literal]^isReady = true;^59^^^^^56^61^isReady = false;^[CLASS] Token Type  [METHOD] reset [RETURN_TYPE] Token   [VARIABLES] Type  EOF  EORECORD  INVALID  TOKEN  type  boolean  isReady  StringBuilder  content  int  INITIAL_TOKEN_LENGTH  