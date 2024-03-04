[BugLab_Variable_Misuse]^return new Float ( value ) ;^73^^^^^72^74^return new Float ( this.value ) ;^[CLASS] MutableFloat  [METHOD] getValue [RETURN_TYPE] Object   [VARIABLES] float  value  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^this.value += value;^129^^^^^128^130^this.value += operand;^[CLASS] MutableFloat  [METHOD] add [RETURN_TYPE] void   float operand [VARIABLES] float  operand  value  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^this.value -= value;^155^^^^^154^156^this.value -= operand;^[CLASS] MutableFloat  [METHOD] subtract [RETURN_TYPE] void   float operand [VARIABLES] float  operand  value  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^this.value += operand;^155^^^^^154^156^this.value -= operand;^[CLASS] MutableFloat  [METHOD] subtract [RETURN_TYPE] void   float operand [VARIABLES] float  operand  value  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^this.value = operand.floatValue (  ) ;^169^^^^^168^170^this.value -= operand.floatValue (  ) ;^[CLASS] MutableFloat  [METHOD] subtract [RETURN_TYPE] void   Number operand [VARIABLES] boolean  float  operand  value  Number  operand  long  serialVersionUID  
[BugLab_Variable_Misuse]^return operand;^198^^^^^197^199^return value;^[CLASS] MutableFloat  [METHOD] floatValue [RETURN_TYPE] float   [VARIABLES] float  operand  value  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return operand;^207^^^^^206^208^return value;^[CLASS] MutableFloat  [METHOD] doubleValue [RETURN_TYPE] double   [VARIABLES] float  operand  value  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return Float.isNaN ( operand ) ;^216^^^^^215^217^return Float.isNaN ( value ) ;^[CLASS] MutableFloat  [METHOD] isNaN [RETURN_TYPE] boolean   [VARIABLES] float  operand  value  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return Float.isInfinite ( operand ) ;^225^^^^^224^226^return Float.isInfinite ( value ) ;^[CLASS] MutableFloat  [METHOD] isInfinite [RETURN_TYPE] boolean   [VARIABLES] float  operand  value  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return  ( obj instanceof MutableFloat ) &&  ( Float.floatToIntBits (  (  ( MutableFloat )  obj ) .value )  == Float.floatToIntBits ( operand )  ) ;^272^273^^^^271^274^return  ( obj instanceof MutableFloat ) &&  ( Float.floatToIntBits (  (  ( MutableFloat )  obj ) .value )  == Float.floatToIntBits ( value )  ) ;^[CLASS] MutableFloat  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  float  operand  value  long  serialVersionUID  
[BugLab_Wrong_Operator]^return  ( obj instanceof MutableFloat ) ||  ( Float.floatToIntBits (  (  ( MutableFloat )  obj ) .value )  == Float.floatToIntBits ( value )  ) ;^272^273^^^^271^274^return  ( obj instanceof MutableFloat ) &&  ( Float.floatToIntBits (  (  ( MutableFloat )  obj ) .value )  == Float.floatToIntBits ( value )  ) ;^[CLASS] MutableFloat  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  float  operand  value  long  serialVersionUID  
[BugLab_Wrong_Operator]^return  ( obj  >>  MutableFloat ) &&  ( Float.floatToIntBits (  (  ( MutableFloat )  obj ) .value )  == Float.floatToIntBits ( value )  ) ;^272^273^^^^271^274^return  ( obj instanceof MutableFloat ) &&  ( Float.floatToIntBits (  (  ( MutableFloat )  obj ) .value )  == Float.floatToIntBits ( value )  ) ;^[CLASS] MutableFloat  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  float  operand  value  long  serialVersionUID  
[BugLab_Wrong_Operator]^return  ( obj instanceof MutableFloat ) &&  ( Float.floatToIntBits (  (  ( MutableFloat )  obj ) .value )  != Float.floatToIntBits ( value )  ) ;^272^273^^^^271^274^return  ( obj instanceof MutableFloat ) &&  ( Float.floatToIntBits (  (  ( MutableFloat )  obj ) .value )  == Float.floatToIntBits ( value )  ) ;^[CLASS] MutableFloat  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  float  operand  value  long  serialVersionUID  
[BugLab_Variable_Misuse]^&&  ( Float.floatToIntBits (  (  ( MutableFloat )  obj ) .value )  == Float.floatToIntBits ( operand )  ) ;^273^^^^^271^274^&&  ( Float.floatToIntBits (  (  ( MutableFloat )  obj ) .value )  == Float.floatToIntBits ( value )  ) ;^[CLASS] MutableFloat  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  float  operand  value  long  serialVersionUID  
[BugLab_Variable_Misuse]^return Float.floatToIntBits ( operand ) ;^283^^^^^282^284^return Float.floatToIntBits ( value ) ;^[CLASS] MutableFloat  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] float  operand  value  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^float anotherVal = value;^295^^^^^293^297^float anotherVal = other.value;^[CLASS] MutableFloat  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  MutableFloat  other  boolean  float  anotherVal  operand  value  long  serialVersionUID  
[BugLab_Argument_Swapping]^float anotherVal = other.value.value;^295^^^^^293^297^float anotherVal = other.value;^[CLASS] MutableFloat  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  MutableFloat  other  boolean  float  anotherVal  operand  value  long  serialVersionUID  
[BugLab_Argument_Swapping]^float anotherVal = other;^295^^^^^293^297^float anotherVal = other.value;^[CLASS] MutableFloat  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  MutableFloat  other  boolean  float  anotherVal  operand  value  long  serialVersionUID  
[BugLab_Variable_Misuse]^return NumberUtils.compare ( value, operand ) ;^296^^^^^293^297^return NumberUtils.compare ( value, anotherVal ) ;^[CLASS] MutableFloat  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  MutableFloat  other  boolean  float  anotherVal  operand  value  long  serialVersionUID  
[BugLab_Variable_Misuse]^return NumberUtils.compare ( operand, anotherVal ) ;^296^^^^^293^297^return NumberUtils.compare ( value, anotherVal ) ;^[CLASS] MutableFloat  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  MutableFloat  other  boolean  float  anotherVal  operand  value  long  serialVersionUID  
[BugLab_Argument_Swapping]^return NumberUtils.compare ( anotherVal, value ) ;^296^^^^^293^297^return NumberUtils.compare ( value, anotherVal ) ;^[CLASS] MutableFloat  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  MutableFloat  other  boolean  float  anotherVal  operand  value  long  serialVersionUID  
[BugLab_Variable_Misuse]^return String.valueOf ( operand ) ;^305^^^^^304^306^return String.valueOf ( value ) ;^[CLASS] MutableFloat  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] float  anotherVal  operand  value  long  serialVersionUID  boolean  
