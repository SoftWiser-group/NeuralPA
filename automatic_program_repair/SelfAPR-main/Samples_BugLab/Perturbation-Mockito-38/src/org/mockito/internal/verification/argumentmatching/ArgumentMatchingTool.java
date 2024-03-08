[BugLab_Variable_Misuse]^if  ( matchers.size (  )  != i )  {^21^^^^^20^37^if  ( matchers.size (  )  != arguments.length )  {^[CLASS] ArgumentMatchingTool  [METHOD] getSuspiciouslyNotMatchingArgsIndexes [RETURN_TYPE] Integer[]   Matcher> matchers Object[] arguments [VARIABLES] boolean  Matcher  m  List  matchers  suspicious  Object[]  arguments  int  i  
[BugLab_Argument_Swapping]^if  ( arguments.size (  )  != matchers.length )  {^21^^^^^20^37^if  ( matchers.size (  )  != arguments.length )  {^[CLASS] ArgumentMatchingTool  [METHOD] getSuspiciouslyNotMatchingArgsIndexes [RETURN_TYPE] Integer[]   Matcher> matchers Object[] arguments [VARIABLES] boolean  Matcher  m  List  matchers  suspicious  Object[]  arguments  int  i  
[BugLab_Argument_Swapping]^if  ( matchers.size (  )  != arguments.length.length )  {^21^^^^^20^37^if  ( matchers.size (  )  != arguments.length )  {^[CLASS] ArgumentMatchingTool  [METHOD] getSuspiciouslyNotMatchingArgsIndexes [RETURN_TYPE] Integer[]   Matcher> matchers Object[] arguments [VARIABLES] boolean  Matcher  m  List  matchers  suspicious  Object[]  arguments  int  i  
[BugLab_Argument_Swapping]^if  ( matchers.size (  )  != arguments )  {^21^^^^^20^37^if  ( matchers.size (  )  != arguments.length )  {^[CLASS] ArgumentMatchingTool  [METHOD] getSuspiciouslyNotMatchingArgsIndexes [RETURN_TYPE] Integer[]   Matcher> matchers Object[] arguments [VARIABLES] boolean  Matcher  m  List  matchers  suspicious  Object[]  arguments  int  i  
[BugLab_Wrong_Operator]^if  ( matchers.size (  )  <= arguments.length )  {^21^^^^^20^37^if  ( matchers.size (  )  != arguments.length )  {^[CLASS] ArgumentMatchingTool  [METHOD] getSuspiciouslyNotMatchingArgsIndexes [RETURN_TYPE] Integer[]   Matcher> matchers Object[] arguments [VARIABLES] boolean  Matcher  m  List  matchers  suspicious  Object[]  arguments  int  i  
[BugLab_Wrong_Literal]^return new Integer[i];^22^^^^^20^37^return new Integer[0];^[CLASS] ArgumentMatchingTool  [METHOD] getSuspiciouslyNotMatchingArgsIndexes [RETURN_TYPE] Integer[]   Matcher> matchers Object[] arguments [VARIABLES] boolean  Matcher  m  List  matchers  suspicious  Object[]  arguments  int  i  
[BugLab_Wrong_Literal]^int i = i;^26^^^^^20^37^int i = 0;^[CLASS] ArgumentMatchingTool  [METHOD] getSuspiciouslyNotMatchingArgsIndexes [RETURN_TYPE] Integer[]   Matcher> matchers Object[] arguments [VARIABLES] boolean  Matcher  m  List  matchers  suspicious  Object[]  arguments  int  i  
[BugLab_Wrong_Operator]^if  ( m instanceof ContainsExtraTypeInformation || !safelyMatches ( m, arguments[i] ) && toStringEquals ( m, arguments[i] ) && ! (  ( ContainsExtraTypeInformation )  m ) .typeMatches ( arguments[i] )  )  {^28^29^30^31^^20^37^if  ( m instanceof ContainsExtraTypeInformation && !safelyMatches ( m, arguments[i] ) && toStringEquals ( m, arguments[i] ) && ! (  ( ContainsExtraTypeInformation )  m ) .typeMatches ( arguments[i] )  )  {^[CLASS] ArgumentMatchingTool  [METHOD] getSuspiciouslyNotMatchingArgsIndexes [RETURN_TYPE] Integer[]   Matcher> matchers Object[] arguments [VARIABLES] boolean  Matcher  m  List  matchers  suspicious  Object[]  arguments  int  i  
[BugLab_Wrong_Operator]^if  ( m  !=  ContainsExtraTypeInformation && !safelyMatches ( m, arguments[i] ) && toStringEquals ( m, arguments[i] ) && ! (  ( ContainsExtraTypeInformation )  m ) .typeMatches ( arguments[i] )  )  {^28^29^30^31^^20^37^if  ( m instanceof ContainsExtraTypeInformation && !safelyMatches ( m, arguments[i] ) && toStringEquals ( m, arguments[i] ) && ! (  ( ContainsExtraTypeInformation )  m ) .typeMatches ( arguments[i] )  )  {^[CLASS] ArgumentMatchingTool  [METHOD] getSuspiciouslyNotMatchingArgsIndexes [RETURN_TYPE] Integer[]   Matcher> matchers Object[] arguments [VARIABLES] boolean  Matcher  m  List  matchers  suspicious  Object[]  arguments  int  i  
[BugLab_Argument_Swapping]^&& !safelyMatches ( arguments, m[i] ) && toStringEquals ( m, arguments[i] ) && ! (  ( ContainsExtraTypeInformation )  m ) .typeMatches ( arguments[i] )  )  {^29^30^31^^^20^37^&& !safelyMatches ( m, arguments[i] ) && toStringEquals ( m, arguments[i] ) && ! (  ( ContainsExtraTypeInformation )  m ) .typeMatches ( arguments[i] )  )  {^[CLASS] ArgumentMatchingTool  [METHOD] getSuspiciouslyNotMatchingArgsIndexes [RETURN_TYPE] Integer[]   Matcher> matchers Object[] arguments [VARIABLES] boolean  Matcher  m  List  matchers  suspicious  Object[]  arguments  int  i  
[BugLab_Argument_Swapping]^&& toStringEquals ( arguments, m[i] ) && ! (  ( ContainsExtraTypeInformation )  m ) .typeMatches ( arguments[i] )  )  {^30^31^^^^20^37^&& toStringEquals ( m, arguments[i] ) && ! (  ( ContainsExtraTypeInformation )  m ) .typeMatches ( arguments[i] )  )  {^[CLASS] ArgumentMatchingTool  [METHOD] getSuspiciouslyNotMatchingArgsIndexes [RETURN_TYPE] Integer[]   Matcher> matchers Object[] arguments [VARIABLES] boolean  Matcher  m  List  matchers  suspicious  Object[]  arguments  int  i  
[BugLab_Wrong_Literal]^return suspicious.toArray ( new Integer[1] ) ;^36^^^^^20^37^return suspicious.toArray ( new Integer[0] ) ;^[CLASS] ArgumentMatchingTool  [METHOD] getSuspiciouslyNotMatchingArgsIndexes [RETURN_TYPE] Integer[]   Matcher> matchers Object[] arguments [VARIABLES] boolean  Matcher  m  List  matchers  suspicious  Object[]  arguments  int  i  
[BugLab_Argument_Swapping]^return arg.matches ( m ) ;^41^^^^^39^45^return m.matches ( arg ) ;^[CLASS] ArgumentMatchingTool  [METHOD] safelyMatches [RETURN_TYPE] boolean   Matcher m Object arg [VARIABLES] boolean  Matcher  m  Throwable  t  Object  arg  
[BugLab_Wrong_Literal]^return true;^43^^^^^39^45^return false;^[CLASS] ArgumentMatchingTool  [METHOD] safelyMatches [RETURN_TYPE] boolean   Matcher m Object arg [VARIABLES] boolean  Matcher  m  Throwable  t  Object  arg  
[BugLab_Argument_Swapping]^return StringDescription.toString ( arg ) .equals ( m == null? "null" : arg.toString (  )  ) ;^48^^^^^47^49^return StringDescription.toString ( m ) .equals ( arg == null? "null" : arg.toString (  )  ) ;^[CLASS] ArgumentMatchingTool  [METHOD] toStringEquals [RETURN_TYPE] boolean   Matcher m Object arg [VARIABLES] boolean  Matcher  m  Object  arg  
[BugLab_Wrong_Operator]^return StringDescription.toString ( m ) .equals ( arg != null? "null" : arg.toString (  )  ) ;^48^^^^^47^49^return StringDescription.toString ( m ) .equals ( arg == null? "null" : arg.toString (  )  ) ;^[CLASS] ArgumentMatchingTool  [METHOD] toStringEquals [RETURN_TYPE] boolean   Matcher m Object arg [VARIABLES] boolean  Matcher  m  Object  arg  