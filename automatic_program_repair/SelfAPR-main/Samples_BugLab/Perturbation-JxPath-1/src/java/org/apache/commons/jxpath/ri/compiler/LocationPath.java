[BugLab_Wrong_Literal]^return false;^41^^^^^39^45^return true;^[CLASS] LocationPath  [METHOD] computeContextDependent [RETURN_TYPE] boolean   [VARIABLES] boolean  absolute  
[BugLab_Wrong_Operator]^if  ( steps == null )  {^50^^^^^47^59^if  ( steps != null )  {^[CLASS] LocationPath  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  Step[]  steps  boolean  absolute  int  i  
[BugLab_Argument_Swapping]^if  ( absolute > 0 || i )  {^52^^^^^47^59^if  ( i > 0 || absolute )  {^[CLASS] LocationPath  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  Step[]  steps  boolean  absolute  int  i  
[BugLab_Wrong_Operator]^if  ( i > 0 && absolute )  {^52^^^^^47^59^if  ( i > 0 || absolute )  {^[CLASS] LocationPath  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  Step[]  steps  boolean  absolute  int  i  
[BugLab_Wrong_Operator]^if  ( i >= 0 || absolute )  {^52^^^^^47^59^if  ( i > 0 || absolute )  {^[CLASS] LocationPath  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  Step[]  steps  boolean  absolute  int  i  
[BugLab_Wrong_Literal]^if  ( i > i || absolute )  {^52^^^^^47^59^if  ( i > 0 || absolute )  {^[CLASS] LocationPath  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  Step[]  steps  boolean  absolute  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= steps.length; i++ )  {^51^^^^^47^59^for  ( int i = 0; i < steps.length; i++ )  {^[CLASS] LocationPath  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  Step[]  steps  boolean  absolute  int  i  
[BugLab_Wrong_Literal]^for  ( int i = i; i < steps.length; i++ )  {^51^^^^^47^59^for  ( int i = 0; i < steps.length; i++ )  {^[CLASS] LocationPath  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  Step[]  steps  boolean  absolute  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i > steps.length; i++ )  {^51^^^^^47^59^for  ( int i = 0; i < steps.length; i++ )  {^[CLASS] LocationPath  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  Step[]  steps  boolean  absolute  int  i  
[BugLab_Wrong_Literal]^for  ( int i = 1; i < steps.length; i++ )  {^51^^^^^47^59^for  ( int i = 0; i < steps.length; i++ )  {^[CLASS] LocationPath  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  Step[]  steps  boolean  absolute  int  i  
[BugLab_Wrong_Literal]^for  ( int i = -1; i < steps.length; i++ )  {^51^^^^^47^59^for  ( int i = 0; i < steps.length; i++ )  {^[CLASS] LocationPath  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  Step[]  steps  boolean  absolute  int  i  
[BugLab_Wrong_Literal]^for  ( int i = ; i < steps.length; i++ )  {^51^^^^^47^59^for  ( int i = 0; i < steps.length; i++ )  {^[CLASS] LocationPath  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  Step[]  steps  boolean  absolute  int  i  
[BugLab_Variable_Misuse]^rootContext = new InitialContext ( rootContext ) ;^68^^^^^61^71^rootContext = new InitialContext ( context ) ;^[CLASS] LocationPath  [METHOD] compute [RETURN_TYPE] Object   EvalContext context [VARIABLES] EvalContext  context  rootContext  boolean  absolute  
[BugLab_Variable_Misuse]^rootContext = rootContext.getRootContext (  ) .getAbsoluteRootContext (  ) ;^65^^^^^61^71^rootContext = context.getRootContext (  ) .getAbsoluteRootContext (  ) ;^[CLASS] LocationPath  [METHOD] compute [RETURN_TYPE] Object   EvalContext context [VARIABLES] EvalContext  context  rootContext  boolean  absolute  
[BugLab_Variable_Misuse]^return evalSteps ( context ) ;^70^^^^^61^71^return evalSteps ( rootContext ) ;^[CLASS] LocationPath  [METHOD] compute [RETURN_TYPE] Object   EvalContext context [VARIABLES] EvalContext  context  rootContext  boolean  absolute  
[BugLab_Variable_Misuse]^rootContext = new InitialContext ( rootContext ) ;^81^^^^^74^84^rootContext = new InitialContext ( context ) ;^[CLASS] LocationPath  [METHOD] computeValue [RETURN_TYPE] Object   EvalContext context [VARIABLES] EvalContext  context  rootContext  boolean  absolute  
[BugLab_Variable_Misuse]^rootContext = rootContext.getRootContext (  ) .getAbsoluteRootContext (  ) ;^78^^^^^74^84^rootContext = context.getRootContext (  ) .getAbsoluteRootContext (  ) ;^[CLASS] LocationPath  [METHOD] computeValue [RETURN_TYPE] Object   EvalContext context [VARIABLES] EvalContext  context  rootContext  boolean  absolute  
[BugLab_Variable_Misuse]^return getSingleNodePointerForSteps ( context ) ;^83^^^^^74^84^return getSingleNodePointerForSteps ( rootContext ) ;^[CLASS] LocationPath  [METHOD] computeValue [RETURN_TYPE] Object   EvalContext context [VARIABLES] EvalContext  context  rootContext  boolean  absolute  