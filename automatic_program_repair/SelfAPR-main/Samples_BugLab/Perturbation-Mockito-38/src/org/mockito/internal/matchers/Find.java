[BugLab_Wrong_Operator]^return actual != null || Pattern.compile ( regex ) .matcher (  ( String )  actual ) .find (  ) ;^21^^^^^20^22^return actual != null && Pattern.compile ( regex ) .matcher (  ( String )  actual ) .find (  ) ;^[CLASS] Find  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  String  regex  boolean  
[BugLab_Wrong_Operator]^return actual == null && Pattern.compile ( regex ) .matcher (  ( String )  actual ) .find (  ) ;^21^^^^^20^22^return actual != null && Pattern.compile ( regex ) .matcher (  ( String )  actual ) .find (  ) ;^[CLASS] Find  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  String  regex  boolean  
[BugLab_Wrong_Operator]^description.appendText ( "find ( \"" + regex.replaceAll ( "\\\\", "\\\\\\\\" )  &&  + "\" ) " ) ;^25^^^^^24^26^description.appendText ( "find ( \"" + regex.replaceAll ( "\\\\", "\\\\\\\\" )  + "\" ) " ) ;^[CLASS] Find  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] String  regex  Description  description  boolean  
[BugLab_Wrong_Operator]^description.appendText ( "find ( \""  &  regex.replaceAll ( "\\\\", "\\\\\\\\" )  + "\" ) " ) ;^25^^^^^24^26^description.appendText ( "find ( \"" + regex.replaceAll ( "\\\\", "\\\\\\\\" )  + "\" ) " ) ;^[CLASS] Find  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] String  regex  Description  description  boolean  
