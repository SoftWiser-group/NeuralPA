[BugLab_Argument_Swapping]^for  ( int i = 0; i < args; i++ )  {^35^^^^^34^41^for  ( int i = 0; i < args.length; i++ )  {^[CLASS] CoreOperationOr  [METHOD] computeValue [RETURN_TYPE] Object   EvalContext context [VARIABLES] boolean  EvalContext  context  int  i  
[BugLab_Argument_Swapping]^for  ( int i = 0; i < args.length.length; i++ )  {^35^^^^^34^41^for  ( int i = 0; i < args.length; i++ )  {^[CLASS] CoreOperationOr  [METHOD] computeValue [RETURN_TYPE] Object   EvalContext context [VARIABLES] boolean  EvalContext  context  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i > args.length; i++ )  {^35^^^^^34^41^for  ( int i = 0; i < args.length; i++ )  {^[CLASS] CoreOperationOr  [METHOD] computeValue [RETURN_TYPE] Object   EvalContext context [VARIABLES] boolean  EvalContext  context  int  i  
[BugLab_Wrong_Literal]^for  ( int i = i; i < args.length; i++ )  {^35^^^^^34^41^for  ( int i = 0; i < args.length; i++ )  {^[CLASS] CoreOperationOr  [METHOD] computeValue [RETURN_TYPE] Object   EvalContext context [VARIABLES] boolean  EvalContext  context  int  i  
[BugLab_Argument_Swapping]^if  ( InfoSetUtil.booleanValue ( context[i].computeValue ( args )  )  )  {^36^^^^^34^41^if  ( InfoSetUtil.booleanValue ( args[i].computeValue ( context )  )  )  {^[CLASS] CoreOperationOr  [METHOD] computeValue [RETURN_TYPE] Object   EvalContext context [VARIABLES] boolean  EvalContext  context  int  i  
[BugLab_Wrong_Literal]^for  ( int i = -1; i < args.length; i++ )  {^35^^^^^34^41^for  ( int i = 0; i < args.length; i++ )  {^[CLASS] CoreOperationOr  [METHOD] computeValue [RETURN_TYPE] Object   EvalContext context [VARIABLES] boolean  EvalContext  context  int  i  
[BugLab_Wrong_Literal]^return false;^48^^^^^47^49^return true;^[CLASS] CoreOperationOr  [METHOD] isSymmetric [RETURN_TYPE] boolean   [VARIABLES] boolean  