[P1_Replace_Type]^private final char suffix;^13^^^^^8^18^private final String suffix;^[CLASS] EndsWith   [VARIABLES] 
[P8_Replace_Mix]^this.suffix =  null;^16^^^^^15^17^this.suffix = suffix;^[CLASS] EndsWith  [METHOD] <init> [RETURN_TYPE] String)   String suffix [VARIABLES] String  suffix  boolean  
[P2_Replace_Operator]^return actual != null ||  (  ( String )  actual ) .endsWith ( suffix ) ;^20^^^^^19^21^return actual != null &&  (  ( String )  actual ) .endsWith ( suffix ) ;^[CLASS] EndsWith  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  String  suffix  boolean  
[P2_Replace_Operator]^return actual == null &&  (  ( String )  actual ) .endsWith ( suffix ) ;^20^^^^^19^21^return actual != null &&  (  ( String )  actual ) .endsWith ( suffix ) ;^[CLASS] EndsWith  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  String  suffix  boolean  
[P5_Replace_Variable]^return suffix != null &&  (  ( String )  actual ) .endsWith ( actual ) ;^20^^^^^19^21^return actual != null &&  (  ( String )  actual ) .endsWith ( suffix ) ;^[CLASS] EndsWith  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  String  suffix  boolean  
[P8_Replace_Mix]^return    (  ( String )  actual ) .endsWith ( suffix ) ;^20^^^^^19^21^return actual != null &&  (  ( String )  actual ) .endsWith ( suffix ) ;^[CLASS] EndsWith  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  String  suffix  boolean  
[P1_Replace_Type]^return actual != null &&  (  ( char )  actual ) .endsWith ( suffix ) ;^20^^^^^19^21^return actual != null &&  (  ( String )  actual ) .endsWith ( suffix ) ;^[CLASS] EndsWith  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  String  suffix  boolean  
[P14_Delete_Statement]^^20^^^^^19^21^return actual != null &&  (  ( String )  actual ) .endsWith ( suffix ) ;^[CLASS] EndsWith  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  String  suffix  boolean  
[P2_Replace_Operator]^description.appendText ( "endsWith ( \""  >=  suffix + "\" ) " ) ;^24^^^^^23^25^description.appendText ( "endsWith ( \"" + suffix + "\" ) " ) ;^[CLASS] EndsWith  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] String  suffix  Description  description  boolean  
[P14_Delete_Statement]^^24^^^^^23^25^description.appendText ( "endsWith ( \"" + suffix + "\" ) " ) ;^[CLASS] EndsWith  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] String  suffix  Description  description  boolean  
