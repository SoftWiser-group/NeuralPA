[BugLab_Argument_Swapping]^set ( b, index ) ;^84^^^^^83^85^set ( index, b ) ;^[CLASS] BooleanList  [METHOD] setBoolean [RETURN_TYPE] void   int index Boolean b [VARIABLES] boolean  Boolean  b  long  serialVersionUID  int  index  
[BugLab_Wrong_Operator]^if  ( obj  ||  BooleanList )  {^95^^^^^94^99^if  ( obj instanceof BooleanList )  {^[CLASS] BooleanList  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] long  serialVersionUID  Object  obj  boolean  
[BugLab_Wrong_Literal]^return true;^98^^^^^94^99^return false;^[CLASS] BooleanList  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] long  serialVersionUID  Object  obj  boolean  
