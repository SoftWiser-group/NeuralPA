[P8_Replace_Mix]^this.wanted =  null;^16^^^^^15^17^this.wanted = wanted;^[CLASS] Same  [METHOD] <init> [RETURN_TYPE] Object)   Object wanted [VARIABLES] Object  wanted  boolean  
[P2_Replace_Operator]^return wanted >= actual;^20^^^^^19^21^return wanted == actual;^[CLASS] Same  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P5_Replace_Variable]^return actual == wanted;^20^^^^^19^21^return wanted == actual;^[CLASS] Same  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P8_Replace_Mix]^return wanted  &&  actual;^20^^^^^19^21^return wanted == actual;^[CLASS] Same  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P14_Delete_Statement]^^24^25^^^^23^29^description.appendText ( "same ( " ) ; appendQuoting ( description ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( "\"" ) ;description.appendText ( "same ( " ) ;^24^^^^^23^29^description.appendText ( "same ( " ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( " ) " ) ;description.appendText ( "same ( " ) ;^24^^^^^23^29^description.appendText ( "same ( " ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( "'" ) ;description.appendText ( "same ( " ) ;^24^^^^^23^29^description.appendText ( "same ( " ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( wanted.toString (  )  ) ;description.appendText ( "same ( " ) ;^24^^^^^23^29^description.appendText ( "same ( " ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P7_Replace_Invocation]^describeTo ( description ) ;^25^^^^^23^29^appendQuoting ( description ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P14_Delete_Statement]^^25^26^^^^23^29^appendQuoting ( description ) ; description.appendText ( wanted.toString (  )  ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P5_Replace_Variable]^description.appendText ( actual.toString (  )  ) ;^26^^^^^23^29^description.appendText ( wanted.toString (  )  ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P14_Delete_Statement]^^26^^^^^23^29^description.appendText ( wanted.toString (  )  ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( "\"" ) ;description.appendText ( wanted.toString (  )  ) ;^26^^^^^23^29^description.appendText ( wanted.toString (  )  ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( " ) " ) ;description.appendText ( wanted.toString (  )  ) ;^26^^^^^23^29^description.appendText ( wanted.toString (  )  ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( "'" ) ;description.appendText ( wanted.toString (  )  ) ;^26^^^^^23^29^description.appendText ( wanted.toString (  )  ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( "same ( " ) ;description.appendText ( wanted.toString (  )  ) ;^26^^^^^23^29^description.appendText ( wanted.toString (  )  ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P7_Replace_Invocation]^describeTo ( description ) ;^27^^^^^23^29^appendQuoting ( description ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P14_Delete_Statement]^^27^28^^^^23^29^appendQuoting ( description ) ; description.appendText ( " ) " ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P14_Delete_Statement]^^28^^^^^23^29^description.appendText ( " ) " ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( "\"" ) ;description.appendText ( " ) " ) ;^28^^^^^23^29^description.appendText ( " ) " ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( "'" ) ;description.appendText ( " ) " ) ;^28^^^^^23^29^description.appendText ( " ) " ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( wanted.toString (  )  ) ;description.appendText ( " ) " ) ;^28^^^^^23^29^description.appendText ( " ) " ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( "same ( " ) ;description.appendText ( " ) " ) ;^28^^^^^23^29^description.appendText ( " ) " ) ;^[CLASS] Same  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P2_Replace_Operator]^if  ( wanted  >>  String )  {^32^^^^^31^37^if  ( wanted instanceof String )  {^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P2_Replace_Operator]^if  ( wanted  >=  String )  {^32^^^^^31^37^if  ( wanted instanceof String )  {^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P5_Replace_Variable]^if  ( actual instanceof String )  {^32^^^^^31^37^if  ( wanted instanceof String )  {^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P6_Replace_Expression]^if  ( wanted instanceof Character )  {^32^^^^^31^37^if  ( wanted instanceof String )  {^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P15_Unwrap_Block]^description.appendText("\"");^32^33^34^35^36^31^37^if  ( wanted instanceof String )  { description.appendText ( "\"" ) ; } else if  ( wanted instanceof Character )  { description.appendText ( "'" ) ; }^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P16_Remove_Block]^^32^33^34^35^36^31^37^if  ( wanted instanceof String )  { description.appendText ( "\"" ) ; } else if  ( wanted instanceof Character )  { description.appendText ( "'" ) ; }^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P2_Replace_Operator]^} else if  ( wanted  >=  Character )  {^34^^^^^31^37^} else if  ( wanted instanceof Character )  {^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P5_Replace_Variable]^} else if  ( actual instanceof Character )  {^34^^^^^31^37^} else if  ( wanted instanceof Character )  {^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P6_Replace_Expression]^} else {^34^^^^^31^37^} else if  ( wanted instanceof Character )  {^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P8_Replace_Mix]^}  if  ( actual instanceof Character )  {^34^^^^^31^37^} else if  ( wanted instanceof Character )  {^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P15_Unwrap_Block]^description.appendText("'");^34^35^36^^^31^37^} else if  ( wanted instanceof Character )  { description.appendText ( "'" ) ; }^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P16_Remove_Block]^^34^35^36^^^31^37^} else if  ( wanted instanceof Character )  { description.appendText ( "'" ) ; }^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P14_Delete_Statement]^^35^^^^^31^37^description.appendText ( "'" ) ;^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( "\"" ) ;description.appendText ( "'" ) ;^35^^^^^31^37^description.appendText ( "'" ) ;^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( " ) " ) ;description.appendText ( "'" ) ;^35^^^^^31^37^description.appendText ( "'" ) ;^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( wanted.toString (  )  ) ;description.appendText ( "'" ) ;^35^^^^^31^37^description.appendText ( "'" ) ;^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( "same ( " ) ;description.appendText ( "'" ) ;^35^^^^^31^37^description.appendText ( "'" ) ;^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P14_Delete_Statement]^^33^^^^^31^37^description.appendText ( "\"" ) ;^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( " ) " ) ;description.appendText ( "\"" ) ;^33^^^^^31^37^description.appendText ( "\"" ) ;^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( "'" ) ;description.appendText ( "\"" ) ;^33^^^^^31^37^description.appendText ( "\"" ) ;^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( wanted.toString (  )  ) ;description.appendText ( "\"" ) ;^33^^^^^31^37^description.appendText ( "\"" ) ;^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P11_Insert_Donor_Statement]^description.appendText ( "same ( " ) ;description.appendText ( "\"" ) ;^33^^^^^31^37^description.appendText ( "\"" ) ;^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P2_Replace_Operator]^} else if  ( wanted  ^  Character )  {^34^^^^^31^37^} else if  ( wanted instanceof Character )  {^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P3_Replace_Literal]^description.appendText ( "" ) ;^35^^^^^31^37^description.appendText ( "'" ) ;^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P3_Replace_Literal]^description.appendText ( """ ) ;^33^^^^^31^37^description.appendText ( "\"" ) ;^[CLASS] Same  [METHOD] appendQuoting [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
