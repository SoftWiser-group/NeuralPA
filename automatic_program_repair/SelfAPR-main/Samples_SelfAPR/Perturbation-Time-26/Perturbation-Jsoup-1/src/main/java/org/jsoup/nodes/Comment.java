[P1_Replace_Type]^private static final char COMMENT_KEY = "comment";^8^^^^^3^13^private static final String COMMENT_KEY = "comment";^[CLASS] Comment   [VARIABLES] 
[P3_Replace_Literal]^private static final String COMMENT_KEY = "omomment";^8^^^^^3^13^private static final String COMMENT_KEY = "comment";^[CLASS] Comment   [VARIABLES] 
[P5_Replace_Variable]^super ( data ) ;^16^^^^^15^18^super ( baseUri ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P14_Delete_Statement]^^16^17^^^^15^18^super ( baseUri ) ; attributes.put ( COMMENT_KEY, data ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P5_Replace_Variable]^attributes.put ( COMMENT_KEY ) ;^17^^^^^15^18^attributes.put ( COMMENT_KEY, data ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P5_Replace_Variable]^attributes.put (  data ) ;^17^^^^^15^18^attributes.put ( COMMENT_KEY, data ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P5_Replace_Variable]^attributes.put ( data, COMMENT_KEY ) ;^17^^^^^15^18^attributes.put ( COMMENT_KEY, data ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P7_Replace_Invocation]^attributes.get ( COMMENT_KEY, data ) ;^17^^^^^15^18^attributes.put ( COMMENT_KEY, data ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P7_Replace_Invocation]^attributes .get ( data )  ;^17^^^^^15^18^attributes.put ( COMMENT_KEY, data ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P8_Replace_Mix]^attributes.get ( baseUri, data ) ;^17^^^^^15^18^attributes.put ( COMMENT_KEY, data ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P14_Delete_Statement]^^17^^^^^15^18^attributes.put ( COMMENT_KEY, data ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P11_Insert_Donor_Statement]^return attributes.get ( COMMENT_KEY ) ;attributes.put ( COMMENT_KEY, data ) ;^17^^^^^15^18^attributes.put ( COMMENT_KEY, data ) ;^[CLASS] Comment  [METHOD] <init> [RETURN_TYPE] String)   String data String baseUri [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P3_Replace_Literal]^return "ommecomment";^21^^^^^20^22^return "#comment";^[CLASS] Comment  [METHOD] nodeName [RETURN_TYPE] String   [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P5_Replace_Variable]^return attributes.get ( data ) ;^29^^^^^28^30^return attributes.get ( COMMENT_KEY ) ;^[CLASS] Comment  [METHOD] getData [RETURN_TYPE] String   [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P5_Replace_Variable]^return COMMENT_KEY.get ( attributes ) ;^29^^^^^28^30^return attributes.get ( COMMENT_KEY ) ;^[CLASS] Comment  [METHOD] getData [RETURN_TYPE] String   [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P8_Replace_Mix]^return attributes .put ( data , COMMENT_KEY )  ;^29^^^^^28^30^return attributes.get ( COMMENT_KEY ) ;^[CLASS] Comment  [METHOD] getData [RETURN_TYPE] String   [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P14_Delete_Statement]^^29^^^^^28^30^return attributes.get ( COMMENT_KEY ) ;^[CLASS] Comment  [METHOD] getData [RETURN_TYPE] String   [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
[P14_Delete_Statement]^^33^34^^^^32^35^indent ( accum ) ; accum.append ( String.format ( "<!--%s-->", getData (  )  )  ) ;^[CLASS] Comment  [METHOD] outerHtml [RETURN_TYPE] void   StringBuilder accum [VARIABLES] StringBuilder  accum  String  COMMENT_KEY  baseUri  data  boolean  
[P3_Replace_Literal]^accum.append ( String.format ( "!--%s!--%s-->", getData (  )  )  ) ;^34^^^^^32^35^accum.append ( String.format ( "<!--%s-->", getData (  )  )  ) ;^[CLASS] Comment  [METHOD] outerHtml [RETURN_TYPE] void   StringBuilder accum [VARIABLES] StringBuilder  accum  String  COMMENT_KEY  baseUri  data  boolean  
[P14_Delete_Statement]^^34^35^^^^32^35^accum.append ( String.format ( "<!--%s-->", getData (  )  )  ) ; }^[CLASS] Comment  [METHOD] outerHtml [RETURN_TYPE] void   StringBuilder accum [VARIABLES] StringBuilder  accum  String  COMMENT_KEY  baseUri  data  boolean  
[P3_Replace_Literal]^accum.append ( String.format ( "<!--%s--><", getData (  )  )  ) ;^34^^^^^32^35^accum.append ( String.format ( "<!--%s-->", getData (  )  )  ) ;^[CLASS] Comment  [METHOD] outerHtml [RETURN_TYPE] void   StringBuilder accum [VARIABLES] StringBuilder  accum  String  COMMENT_KEY  baseUri  data  boolean  
[P14_Delete_Statement]^^38^^^^^37^39^return outerHtml (  ) ;^[CLASS] Comment  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] String  COMMENT_KEY  baseUri  data  boolean  
