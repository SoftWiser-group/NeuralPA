[BugLab_Argument_Swapping]^return expectedType.equals ( typeOfValue ) ;^25^^^^^24^26^return typeOfValue.equals ( expectedType ) ;^[CLASS] Util  [METHOD] isAssignableFrom [RETURN_TYPE] boolean   Type typeOfValue Type expectedType [VARIABLES] Type  expectedType  typeOfValue  boolean  
[BugLab_Wrong_Literal]^boolean first = false;^30^^^^^28^41^boolean first = true;^[CLASS] Util  [METHOD] toStringMapKeys [RETURN_TYPE] String   String, ?> map [VARIABLES] Entry  entry  boolean  first  StringBuilder  sb  Map  map  
[BugLab_Wrong_Literal]^first = true;^33^^^^^28^41^first = false;^[CLASS] Util  [METHOD] toStringMapKeys [RETURN_TYPE] String   String, ?> map [VARIABLES] Entry  entry  boolean  first  StringBuilder  sb  Map  map  
[BugLab_Variable_Misuse]^sb.append ( null.getKey (  )  ) ;^37^^^^^28^41^sb.append ( entry.getKey (  )  ) ;^[CLASS] Util  [METHOD] toStringMapKeys [RETURN_TYPE] String   String, ?> map [VARIABLES] Entry  entry  boolean  first  StringBuilder  sb  Map  map  
[BugLab_Wrong_Literal]^boolean first = false;^45^^^^^43^58^boolean first = true;^[CLASS] Util  [METHOD] toStringMapOfTypes [RETURN_TYPE] String   Type> map [VARIABLES] Entry  entry  boolean  first  StringBuilder  sb  Class  clazz  Map  map  
[BugLab_Wrong_Literal]^first = true;^48^^^^^43^58^first = false;^[CLASS] Util  [METHOD] toStringMapOfTypes [RETURN_TYPE] String   Type> map [VARIABLES] Entry  entry  boolean  first  StringBuilder  sb  Class  clazz  Map  map  
[BugLab_Wrong_Literal]^boolean first = false;^62^^^^^60^73^boolean first = true;^[CLASS] Util  [METHOD] toStringMap [RETURN_TYPE] String   Object> map [VARIABLES] Entry  entry  boolean  first  StringBuilder  sb  Map  map  
[BugLab_Wrong_Literal]^first = true;^65^^^^^60^73^first = false;^[CLASS] Util  [METHOD] toStringMap [RETURN_TYPE] String   Object> map [VARIABLES] Entry  entry  boolean  first  StringBuilder  sb  Map  map  
