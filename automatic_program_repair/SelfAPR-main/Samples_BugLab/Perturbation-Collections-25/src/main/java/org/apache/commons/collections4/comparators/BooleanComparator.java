[BugLab_Wrong_Literal]^private static final BooleanComparator TRUE_FIRST = new BooleanComparator ( false ) ;^39^^^^^34^44^private static final BooleanComparator TRUE_FIRST = new BooleanComparator ( true ) ;^[CLASS] BooleanComparator   [VARIABLES] 
[BugLab_Wrong_Literal]^private static final BooleanComparator FALSE_FIRST = new BooleanComparator ( true ) ;^42^^^^^37^47^private static final BooleanComparator FALSE_FIRST = new BooleanComparator ( false ) ;^[CLASS] BooleanComparator   [VARIABLES] 
[BugLab_Wrong_Literal]^private boolean falseFirst = false;^45^^^^^40^50^private boolean trueFirst = false;^[CLASS] BooleanComparator   [VARIABLES] 
[BugLab_Wrong_Literal]^this ( true ) ;^110^^^^^109^111^this ( false ) ;^[CLASS] BooleanComparator  [METHOD] <init> [RETURN_TYPE] BooleanComparator()   [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  long  serialVersionUID  
[BugLab_Variable_Misuse]^return FALSE_FIRST;^61^^^^^60^62^return TRUE_FIRST;^[CLASS] BooleanComparator  [METHOD] getTrueFirstComparator [RETURN_TYPE] BooleanComparator   [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  long  serialVersionUID  
[BugLab_Argument_Swapping]^return TRUE_FIRST ? trueFirst : FALSE_FIRST;^97^^^^^96^98^return trueFirst ? TRUE_FIRST : FALSE_FIRST;^[CLASS] BooleanComparator  [METHOD] booleanComparator [RETURN_TYPE] BooleanComparator   final boolean trueFirst [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  long  serialVersionUID  
[BugLab_Argument_Swapping]^return trueFirst ? FALSE_FIRST : TRUE_FIRST;^97^^^^^96^98^return trueFirst ? TRUE_FIRST : FALSE_FIRST;^[CLASS] BooleanComparator  [METHOD] booleanComparator [RETURN_TYPE] BooleanComparator   final boolean trueFirst [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  long  serialVersionUID  
[BugLab_Variable_Misuse]^final boolean v1 = b2.booleanValue (  ) ;^138^^^^^137^142^final boolean v1 = b1.booleanValue (  ) ;^[CLASS] BooleanComparator  [METHOD] compare [RETURN_TYPE] int   Boolean b1 Boolean b2 [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  Boolean  b1  b2  long  serialVersionUID  
[BugLab_Variable_Misuse]^final boolean v2 = b1.booleanValue (  ) ;^139^^^^^137^142^final boolean v2 = b2.booleanValue (  ) ;^[CLASS] BooleanComparator  [METHOD] compare [RETURN_TYPE] int   Boolean b1 Boolean b2 [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  Boolean  b1  b2  long  serialVersionUID  
[BugLab_Variable_Misuse]^return  ( trueFirst ^ v2 )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 0;^141^^^^^137^142^return  ( v1 ^ v2 )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 0;^[CLASS] BooleanComparator  [METHOD] compare [RETURN_TYPE] int   Boolean b1 Boolean b2 [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  Boolean  b1  b2  long  serialVersionUID  
[BugLab_Variable_Misuse]^return  ( v1 ^ trueFirst )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 0;^141^^^^^137^142^return  ( v1 ^ v2 )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 0;^[CLASS] BooleanComparator  [METHOD] compare [RETURN_TYPE] int   Boolean b1 Boolean b2 [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  Boolean  b1  b2  long  serialVersionUID  
[BugLab_Argument_Swapping]^return  ( trueFirst ^ v2 )  ?  (   ( v1 ^ v1 )  ? 1 : -1  )  : 0;^141^^^^^137^142^return  ( v1 ^ v2 )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 0;^[CLASS] BooleanComparator  [METHOD] compare [RETURN_TYPE] int   Boolean b1 Boolean b2 [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  Boolean  b1  b2  long  serialVersionUID  
[BugLab_Argument_Swapping]^return  ( v1 ^ trueFirst )  ?  (   ( v1 ^ v2 )  ? 1 : -1  )  : 0;^141^^^^^137^142^return  ( v1 ^ v2 )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 0;^[CLASS] BooleanComparator  [METHOD] compare [RETURN_TYPE] int   Boolean b1 Boolean b2 [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  Boolean  b1  b2  long  serialVersionUID  
[BugLab_Wrong_Operator]^return  ( v1 & v2 )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 0;^141^^^^^137^142^return  ( v1 ^ v2 )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 0;^[CLASS] BooleanComparator  [METHOD] compare [RETURN_TYPE] int   Boolean b1 Boolean b2 [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  Boolean  b1  b2  long  serialVersionUID  
[BugLab_Wrong_Operator]^return  ( v1 ^ v2 )  ?  (   ( v1 & trueFirst )  ? 1 : -1  )  : 0;^141^^^^^137^142^return  ( v1 ^ v2 )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 0;^[CLASS] BooleanComparator  [METHOD] compare [RETURN_TYPE] int   Boolean b1 Boolean b2 [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  Boolean  b1  b2  long  serialVersionUID  
[BugLab_Wrong_Literal]^return  ( v2 ^ v2 )  ?  (   ( v2 ^ trueFirst )  ? 2 : -2  )  : 0;^141^^^^^137^142^return  ( v1 ^ v2 )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 0;^[CLASS] BooleanComparator  [METHOD] compare [RETURN_TYPE] int   Boolean b1 Boolean b2 [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  Boolean  b1  b2  long  serialVersionUID  
[BugLab_Wrong_Literal]^return  ( v0 ^ v2 )  ?  (   ( v0 ^ trueFirst )  ? 0 : -0  )  : 0;^141^^^^^137^142^return  ( v1 ^ v2 )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 0;^[CLASS] BooleanComparator  [METHOD] compare [RETURN_TYPE] int   Boolean b1 Boolean b2 [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  Boolean  b1  b2  long  serialVersionUID  
[BugLab_Wrong_Literal]^return  ( v1 ^ v2 )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 1;^141^^^^^137^142^return  ( v1 ^ v2 )  ?  (   ( v1 ^ trueFirst )  ? 1 : -1  )  : 0;^[CLASS] BooleanComparator  [METHOD] compare [RETURN_TYPE] int   Boolean b1 Boolean b2 [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  Boolean  b1  b2  long  serialVersionUID  
[BugLab_Variable_Misuse]^return v2 ? -1 * hash : hash;^154^^^^^152^155^return trueFirst ? -1 * hash : hash;^[CLASS] BooleanComparator  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  long  serialVersionUID  int  hash  
[BugLab_Argument_Swapping]^return hash ? -1 * trueFirst : hash;^154^^^^^152^155^return trueFirst ? -1 * hash : hash;^[CLASS] BooleanComparator  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  long  serialVersionUID  int  hash  
[BugLab_Wrong_Operator]^return trueFirst ? -1 + hash : hash;^154^^^^^152^155^return trueFirst ? -1 * hash : hash;^[CLASS] BooleanComparator  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  long  serialVersionUID  int  hash  
[BugLab_Wrong_Literal]^return trueFirst ? -0 * hash : hash;^154^^^^^152^155^return trueFirst ? -1 * hash : hash;^[CLASS] BooleanComparator  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  long  serialVersionUID  int  hash  
[BugLab_Variable_Misuse]^return  ( this == object )  || (  ( object instanceof BooleanComparator )  && ( this.v2 ==  (  ( BooleanComparator ) object ) .trueFirst )  ) ;^171^172^173^^^170^174^return  ( this == object )  || (  ( object instanceof BooleanComparator )  && ( this.trueFirst ==  (  ( BooleanComparator ) object ) .trueFirst )  ) ;^[CLASS] BooleanComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  long  serialVersionUID  
[BugLab_Argument_Swapping]^return  ( this == this.trueFirst )  || (  ( object instanceof BooleanComparator )  && ( object ==  (  ( BooleanComparator ) object ) .trueFirst )  ) ;^171^172^173^^^170^174^return  ( this == object )  || (  ( object instanceof BooleanComparator )  && ( this.trueFirst ==  (  ( BooleanComparator ) object ) .trueFirst )  ) ;^[CLASS] BooleanComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  long  serialVersionUID  
[BugLab_Wrong_Operator]^return  ( this == object )  && (  ( object instanceof BooleanComparator )  && ( this.trueFirst ==  (  ( BooleanComparator ) object ) .trueFirst )  ) ;^171^172^173^^^170^174^return  ( this == object )  || (  ( object instanceof BooleanComparator )  && ( this.trueFirst ==  (  ( BooleanComparator ) object ) .trueFirst )  ) ;^[CLASS] BooleanComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  long  serialVersionUID  
[BugLab_Wrong_Operator]^return  ( this != object )  || (  ( object instanceof BooleanComparator )  && ( this.trueFirst ==  (  ( BooleanComparator ) object ) .trueFirst )  ) ;^171^172^173^^^170^174^return  ( this == object )  || (  ( object instanceof BooleanComparator )  && ( this.trueFirst ==  (  ( BooleanComparator ) object ) .trueFirst )  ) ;^[CLASS] BooleanComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  long  serialVersionUID  
[BugLab_Wrong_Operator]^return  ( this == object )  || (  ( object instanceof BooleanComparator )  || ( this.trueFirst ==  (  ( BooleanComparator ) object ) .trueFirst )  ) ;^171^172^173^^^170^174^return  ( this == object )  || (  ( object instanceof BooleanComparator )  && ( this.trueFirst ==  (  ( BooleanComparator ) object ) .trueFirst )  ) ;^[CLASS] BooleanComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  long  serialVersionUID  
[BugLab_Wrong_Operator]^return  ( this == object )  || (  ( object  <  BooleanComparator )  && ( this.trueFirst ==  (  ( BooleanComparator ) object ) .trueFirst )  ) ;^171^172^173^^^170^174^return  ( this == object )  || (  ( object instanceof BooleanComparator )  && ( this.trueFirst ==  (  ( BooleanComparator ) object ) .trueFirst )  ) ;^[CLASS] BooleanComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  long  serialVersionUID  
[BugLab_Variable_Misuse]^return v2;^188^^^^^187^189^return trueFirst;^[CLASS] BooleanComparator  [METHOD] sortsTrueFirst [RETURN_TYPE] boolean   [VARIABLES] BooleanComparator  FALSE_FIRST  TRUE_FIRST  boolean  trueFirst  v1  v2  long  serialVersionUID  
