[BugLab_Variable_Misuse]^super ( data ) ;^16^^^^^15^18^super ( baseUri ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[BugLab_Variable_Misuse]^attributes.put ( COMMENT_KEY, baseUri ) ;^17^^^^^15^18^attributes.put ( COMMENT_KEY, data ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[BugLab_Argument_Swapping]^attributes.put ( data, COMMENT_KEY ) ;^17^^^^^15^18^attributes.put ( COMMENT_KEY, data ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[BugLab_Variable_Misuse]^return attributes.get ( data ) ;^29^^^^^28^30^return attributes.get ( COMMENT_KEY ) ;^[CLASS] Comment  [METHOD] getData [RETURN_TYPE] String   [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[BugLab_Argument_Swapping]^return COMMENT_KEY.get ( attributes ) ;^29^^^^^28^30^return attributes.get ( COMMENT_KEY ) ;^[CLASS] Comment  [METHOD] getData [RETURN_TYPE] String   [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
