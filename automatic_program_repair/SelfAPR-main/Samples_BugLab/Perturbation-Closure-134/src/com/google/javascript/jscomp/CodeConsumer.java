[BugLab_Wrong_Literal]^boolean statementNeedsEnded = true;^33^^^^^28^38^boolean statementNeedsEnded = false;^[CLASS] CodeConsumer   [VARIABLES] 
[BugLab_Wrong_Literal]^boolean statementStarted = true;^34^^^^^29^39^boolean statementStarted = false;^[CLASS] CodeConsumer   [VARIABLES] 
[BugLab_Wrong_Literal]^boolean sawFunction = true;^35^^^^^30^40^boolean sawFunction = false;^[CLASS] CodeConsumer   [VARIABLES] 
[BugLab_Wrong_Literal]^return false;^82^^^^^81^83^return true;^[CLASS] CodeConsumer  [METHOD] continueProcessing [RETURN_TYPE] boolean   [VARIABLES] boolean  sawFunction  statementNeedsEnded  statementStarted  
[BugLab_Variable_Misuse]^if  ( statementStarted )  {^130^^^^^129^138^if  ( statementNeedsEnded )  {^[CLASS] CodeConsumer  [METHOD] beginBlock [RETURN_TYPE] void   [VARIABLES] boolean  sawFunction  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Literal]^statementNeedsEnded = true;^137^^^^^129^138^statementNeedsEnded = false;^[CLASS] CodeConsumer  [METHOD] beginBlock [RETURN_TYPE] void   [VARIABLES] boolean  sawFunction  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Literal]^endBlock ( true ) ;^141^^^^^140^142^endBlock ( false ) ;^[CLASS] CodeConsumer  [METHOD] endBlock [RETURN_TYPE] void   [VARIABLES] boolean  sawFunction  statementNeedsEnded  statementStarted  
[BugLab_Variable_Misuse]^if  ( statementStarted )  {^146^^^^^144^150^if  ( statementContext )  {^[CLASS] CodeConsumer  [METHOD] endBlock [RETURN_TYPE] void   boolean statementContext [VARIABLES] boolean  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Literal]^statementNeedsEnded = true;^149^^^^^144^150^statementNeedsEnded = false;^[CLASS] CodeConsumer  [METHOD] endBlock [RETURN_TYPE] void   boolean statementContext [VARIABLES] boolean  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Literal]^endStatement ( true ) ;^164^^^^^163^165^endStatement ( false ) ;^[CLASS] CodeConsumer  [METHOD] endStatement [RETURN_TYPE] void   [VARIABLES] boolean  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Variable_Misuse]^if  ( statementNeedsEnded )  {^168^^^^^167^175^if  ( needSemiColon )  {^[CLASS] CodeConsumer  [METHOD] endStatement [RETURN_TYPE] void   boolean needSemiColon [VARIABLES] boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Variable_Misuse]^} else if  ( statementContext )  {^172^^^^^167^175^} else if  ( statementStarted )  {^[CLASS] CodeConsumer  [METHOD] endStatement [RETURN_TYPE] void   boolean needSemiColon [VARIABLES] boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Literal]^statementNeedsEnded = false;^173^^^^^167^175^statementNeedsEnded = true;^[CLASS] CodeConsumer  [METHOD] endStatement [RETURN_TYPE] void   boolean needSemiColon [VARIABLES] boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Literal]^statementNeedsEnded = true;^171^^^^^167^175^statementNeedsEnded = false;^[CLASS] CodeConsumer  [METHOD] endStatement [RETURN_TYPE] void   boolean needSemiColon [VARIABLES] boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Variable_Misuse]^} else if  ( statementNeedsEnded )  {^172^^^^^167^175^} else if  ( statementStarted )  {^[CLASS] CodeConsumer  [METHOD] endStatement [RETURN_TYPE] void   boolean needSemiColon [VARIABLES] boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Variable_Misuse]^if  ( statementStarted )  {^183^^^^^181^190^if  ( statementNeedsEnded )  {^[CLASS] CodeConsumer  [METHOD] maybeEndStatement [RETURN_TYPE] void   [VARIABLES] boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Literal]^statementNeedsEnded = true;^187^^^^^181^190^statementNeedsEnded = false;^[CLASS] CodeConsumer  [METHOD] maybeEndStatement [RETURN_TYPE] void   [VARIABLES] boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Literal]^statementStarted = false;^189^^^^^181^190^statementStarted = true;^[CLASS] CodeConsumer  [METHOD] maybeEndStatement [RETURN_TYPE] void   [VARIABLES] boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Literal]^endFunction ( true ) ;^193^^^^^192^194^endFunction ( false ) ;^[CLASS] CodeConsumer  [METHOD] endFunction [RETURN_TYPE] void   [VARIABLES] boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Literal]^sawFunction = false;^197^^^^^196^201^sawFunction = true;^[CLASS] CodeConsumer  [METHOD] endFunction [RETURN_TYPE] void   boolean statementContext [VARIABLES] boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Variable_Misuse]^if  ( statementStarted )  {^198^^^^^196^201^if  ( statementContext )  {^[CLASS] CodeConsumer  [METHOD] endFunction [RETURN_TYPE] void   boolean statementContext [VARIABLES] boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Operator]^if  ( newcode.length (  )  >= 0 )  {^213^^^^^210^226^if  ( newcode.length (  )  == 0 )  {^[CLASS] CodeConsumer  [METHOD] add [RETURN_TYPE] void   String newcode [VARIABLES] char  c  boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  newcode  
[BugLab_Wrong_Literal]^if  ( newcode.length (  )  == 1 )  {^213^^^^^210^226^if  ( newcode.length (  )  == 0 )  {^[CLASS] CodeConsumer  [METHOD] add [RETURN_TYPE] void   String newcode [VARIABLES] char  c  boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  newcode  
[BugLab_Wrong_Literal]^char c = newcode.charAt (  ) ;^217^^^^^210^226^char c = newcode.charAt ( 0 ) ;^[CLASS] CodeConsumer  [METHOD] add [RETURN_TYPE] void   String newcode [VARIABLES] char  c  boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  newcode  
[BugLab_Wrong_Literal]^char c = newcode.charAt ( 1 ) ;^217^^^^^210^226^char c = newcode.charAt ( 0 ) ;^[CLASS] CodeConsumer  [METHOD] add [RETURN_TYPE] void   String newcode [VARIABLES] char  c  boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  newcode  
[BugLab_Wrong_Operator]^if  (  ( isWordChar ( c )  || c == '\\' )  || isWordChar ( getLastChar (  )  )  )  {^218^219^^^^210^226^if  (  ( isWordChar ( c )  || c == '\\' )  && isWordChar ( getLastChar (  )  )  )  {^[CLASS] CodeConsumer  [METHOD] add [RETURN_TYPE] void   String newcode [VARIABLES] char  c  boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  newcode  
[BugLab_Wrong_Operator]^if  (  ( isWordChar ( c )  && c == '\\' )  && isWordChar ( getLastChar (  )  )  )  {^218^219^^^^210^226^if  (  ( isWordChar ( c )  || c == '\\' )  && isWordChar ( getLastChar (  )  )  )  {^[CLASS] CodeConsumer  [METHOD] add [RETURN_TYPE] void   String newcode [VARIABLES] char  c  boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  newcode  
[BugLab_Wrong_Operator]^if  (  ( isWordChar ( c )  || c < '\\' )  && isWordChar ( getLastChar (  )  )  )  {^218^219^^^^210^226^if  (  ( isWordChar ( c )  || c == '\\' )  && isWordChar ( getLastChar (  )  )  )  {^[CLASS] CodeConsumer  [METHOD] add [RETURN_TYPE] void   String newcode [VARIABLES] char  c  boolean  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  newcode  
[BugLab_Argument_Swapping]^if  (  ( prev == '+' || first == '-' )  && first == first )  {^238^^^^^232^260^if  (  ( first == '+' || first == '-' )  && prev == first )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^if  (  ( first == '+' || first == '-' )  || prev == first )  {^238^^^^^232^260^if  (  ( first == '+' || first == '-' )  && prev == first )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^if  (  ( first == '+' && first == '-' )  && prev == first )  {^238^^^^^232^260^if  (  ( first == '+' || first == '-' )  && prev == first )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^if  (  ( first >= '+' || first == '-' )  && prev == first )  {^238^^^^^232^260^if  (  ( first == '+' || first == '-' )  && prev == first )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^if  (  ( first == '+' || first > '-' )  && prev == first )  {^238^^^^^232^260^if  (  ( first == '+' || first == '-' )  && prev == first )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^if  (  ( first == '+' || first == '-' )  && prev != first )  {^238^^^^^232^260^if  (  ( first == '+' || first == '-' )  && prev == first )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^if  (  ( first != '+' || first == '-' )  && prev == first )  {^238^^^^^232^260^if  (  ( first == '+' || first == '-' )  && prev == first )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Argument_Swapping]^} else if  ( Character.isLetter ( prev )  && isWordChar ( first )  )  {^242^243^^^^232^260^} else if  ( Character.isLetter ( first )  && isWordChar ( prev )  )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^} else if  ( Character.isLetter ( first )  || isWordChar ( prev )  )  {^242^243^^^^232^260^} else if  ( Character.isLetter ( first )  && isWordChar ( prev )  )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Argument_Swapping]^} else if  ( first == '-' && prev == '>' )  {^246^^^^^232^260^} else if  ( prev == '-' && first == '>' )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^} else if  ( prev == '-' || first == '>' )  {^246^^^^^232^260^} else if  ( prev == '-' && first == '>' )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^} else if  ( prev != '-' && first == '>' )  {^246^^^^^232^260^} else if  ( prev == '-' && first == '>' )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^} else if  ( prev == '-' && first <= '>' )  {^246^^^^^232^260^} else if  ( prev == '-' && first == '>' )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Variable_Misuse]^} else if  ( Character.isLetter ( prev )  && isWordChar ( prev )  )  {^242^243^^^^232^260^} else if  ( Character.isLetter ( first )  && isWordChar ( prev )  )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Variable_Misuse]^isWordChar ( first )  )  {^243^^^^^232^260^isWordChar ( prev )  )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^} else if  ( prev <= '-' && first == '>' )  {^246^^^^^232^260^} else if  ( prev == '-' && first == '>' )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^} else if  ( prev == '-' && first != '>' )  {^246^^^^^232^260^} else if  ( prev == '-' && first == '>' )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Wrong_Operator]^} else if  ( prev >= '-' && first == '>' )  {^246^^^^^232^260^} else if  ( prev == '-' && first == '>' )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Variable_Misuse]^appendOp ( op, statementStarted ) ;^252^^^^^232^260^appendOp ( op, binOp ) ;^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Argument_Swapping]^appendOp ( binOp, op ) ;^252^^^^^232^260^appendOp ( op, binOp ) ;^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Variable_Misuse]^if  ( statementNeedsEnded )  {^257^^^^^232^260^if  ( binOp )  {^[CLASS] CodeConsumer  [METHOD] addOp [RETURN_TYPE] void   String op boolean binOp [VARIABLES] char  first  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  String  op  
[BugLab_Argument_Swapping]^if  ( prev < 0 && x == '-' )  {^266^^^^^262^288^if  ( x < 0 && prev == '-' )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^if  ( x < 0 || prev == '-' )  {^266^^^^^262^288^if  ( x < 0 && prev == '-' )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^if  ( x == 0 && prev == '-' )  {^266^^^^^262^288^if  ( x < 0 && prev == '-' )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^if  ( x < 0 && prev >= '-' )  {^266^^^^^262^288^if  ( x < 0 && prev == '-' )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^if  ( x < exp && prev == '-' )  {^266^^^^^262^288^if  ( x < 0 && prev == '-' )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Variable_Misuse]^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == mantissa )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Argument_Swapping]^while  ( value / 10 * Math.pow ( 10, exp + 1 )  == mantissa )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Argument_Swapping]^while  ( exp / 10 * Math.pow ( 10, mantissa + 1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  >= value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while + ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa - 10 * Math.pow ( 10, exp + 1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa / 10 * Math.pow ( 10, exp  ||  1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^while  ( mantissa / exp * Math.pow ( exp, exp + 1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^while  ( mantissa / 10 * Math.pow ( 10, exp  )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^mantissa /= exp;^276^^^^^262^288^mantissa /= 10;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^if  (  ( long )  x != x )  {^270^^^^^262^288^if  (  ( long )  x == x )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Argument_Swapping]^while  ( mantissa / 10 * Math.pow ( 10, value + 1 )  == exp )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  != value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while / ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa / 10 * Math.pow ( 10, exp  |  1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^mantissa /= 9;^276^^^^^262^288^mantissa /= 10;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^if  ( x > 100 )  {^274^^^^^262^288^if  ( x >= 100 )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^if  ( x >= exp )  {^274^^^^^262^288^if  ( x >= 100 )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^if  ( x >= 90 )  {^274^^^^^262^288^if  ( x >= 100 )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa * 10 * Math.pow ( 10, exp + 1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa / 10 * Math.pow ( 10, exp  >>  1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^while  ( mantissa / exp0 * Math.pow ( exp0, exp + exp )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^while  ( mantissa / 9 * Math.pow ( 9, exp + 1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa / 10 * Math.pow ( 10, exp  <<  1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^if  ( exp < 2 )  {^280^^^^^262^288^if  ( exp > 2 )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Variable_Misuse]^add ( Long.toString ( mantissa )  ) ;^283^^^^^280^284^add ( Long.toString ( value )  ) ;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Variable_Misuse]^add ( Long.toString ( value )  + "E" + Integer.toString ( exp )  ) ;^281^^^^^262^288^add ( Long.toString ( mantissa )  + "E" + Integer.toString ( exp )  ) ;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Argument_Swapping]^add ( Long.toString ( exp )  + "E" + Integer.toString ( mantissa )  ) ;^281^^^^^262^288^add ( Long.toString ( mantissa )  + "E" + Integer.toString ( exp )  ) ;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^add ( Long.toString ( mantissa )  &&  + "E" + Integer.toString ( exp )  ) ;^281^^^^^262^288^add ( Long.toString ( mantissa )  + "E" + Integer.toString ( exp )  ) ;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^add ( Long.toString ( mantissa )   ^  "E" + Integer.toString ( exp )  ) ;^281^^^^^262^288^add ( Long.toString ( mantissa )  + "E" + Integer.toString ( exp )  ) ;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^int exp = exp;^273^^^^^262^288^int exp = 0;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa / 10 * Math.pow ( 10, exp  !=  1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^add ( Long.toString ( mantissa )  >>  + "E" + Integer.toString ( exp )  ) ;^281^^^^^262^288^add ( Long.toString ( mantissa )  + "E" + Integer.toString ( exp )  ) ;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^add ( Long.toString ( mantissa )   <<  "E" + Integer.toString ( exp )  ) ;^281^^^^^262^288^add ( Long.toString ( mantissa )  + "E" + Integer.toString ( exp )  ) ;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Variable_Misuse]^add ( Long.toString ( mantissa )  ) ;^283^^^^^262^288^add ( Long.toString ( value )  ) ;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^if  ( x >= 101 )  {^274^^^^^262^288^if  ( x >= 100 )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^if  ( x >= exp0 )  {^274^^^^^262^288^if  ( x >= 100 )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^if  ( x >= 00 )  {^274^^^^^262^288^if  ( x >= 100 )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Variable_Misuse]^while  ( value / 10 * Math.pow ( 10, exp + 1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  <= value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa / 10 * Math.pow ( 10, exp  >  1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^mantissa /= 11;^276^^^^^262^288^mantissa /= 10;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa / 10 * Math.pow ( 10, exp   instanceof   1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^if  ( exp >= 2 )  {^280^^^^^262^288^if  ( exp > 2 )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Literal]^if  ( exp > 3 )  {^280^^^^^262^288^if  ( exp > 2 )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^add ( Long.toString ( mantissa )  ||  + "E" + Integer.toString ( exp )  ) ;^281^^^^^262^288^add ( Long.toString ( mantissa )  + "E" + Integer.toString ( exp )  ) ;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^add ( Long.toString ( mantissa )   >=  "E" + Integer.toString ( exp )  ) ;^281^^^^^262^288^add ( Long.toString ( mantissa )  + "E" + Integer.toString ( exp )  ) ;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^while  ( mantissa / 10 * Math.pow ( 10, exp  <  1 )  == value )  {^275^^^^^262^288^while  ( mantissa / 10 * Math.pow ( 10, exp + 1 )  == value )  {^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^add ( Long.toString ( mantissa )  <<  + "E" + Integer.toString ( exp )  ) ;^281^^^^^262^288^add ( Long.toString ( mantissa )  + "E" + Integer.toString ( exp )  ) ;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^add ( Long.toString ( mantissa )   >>  "E" + Integer.toString ( exp )  ) ;^281^^^^^262^288^add ( Long.toString ( mantissa )  + "E" + Integer.toString ( exp )  ) ;^[CLASS] CodeConsumer  [METHOD] addNumber [RETURN_TYPE] void   double x [VARIABLES] char  prev  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  double  x  long  mantissa  value  int  exp  
[BugLab_Wrong_Operator]^return  ( ch == '_' && ch == '$' || Character.isLetterOrDigit ( ch )  ) ;^291^292^293^^^290^294^return  ( ch == '_' || ch == '$' || Character.isLetterOrDigit ( ch )  ) ;^[CLASS] CodeConsumer  [METHOD] isWordChar [RETURN_TYPE] boolean   char ch [VARIABLES] char  ch  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Operator]^return  ( ch >= '_' || ch == '$' || Character.isLetterOrDigit ( ch )  ) ;^291^292^293^^^290^294^return  ( ch == '_' || ch == '$' || Character.isLetterOrDigit ( ch )  ) ;^[CLASS] CodeConsumer  [METHOD] isWordChar [RETURN_TYPE] boolean   char ch [VARIABLES] char  ch  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
[BugLab_Wrong_Operator]^return  ( ch == '_' || ch != '$' || Character.isLetterOrDigit ( ch )  ) ;^291^292^293^^^290^294^return  ( ch == '_' || ch == '$' || Character.isLetterOrDigit ( ch )  ) ;^[CLASS] CodeConsumer  [METHOD] isWordChar [RETURN_TYPE] boolean   char ch [VARIABLES] char  ch  boolean  binOp  needSemiColon  sawFunction  statementContext  statementNeedsEnded  statementStarted  
