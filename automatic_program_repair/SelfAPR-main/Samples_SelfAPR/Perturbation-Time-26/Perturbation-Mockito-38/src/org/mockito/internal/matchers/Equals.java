[P8_Replace_Mix]^private  Object wanted;^13^^^^^8^18^private final Object wanted;^[CLASS] Equals 1   [VARIABLES] 
[P8_Replace_Mix]^this.wanted =  null;^16^^^^^15^17^this.wanted = wanted;^[CLASS] Equals 1  [METHOD] <init> [RETURN_TYPE] Object)   Object wanted [VARIABLES] Object  wanted  boolean  
[P2_Replace_Operator]^if  ( this.wanted != null )  {^20^^^^^19^24^if  ( this.wanted == null )  {^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P5_Replace_Variable]^if  ( wanted == null )  {^20^^^^^19^24^if  ( this.wanted == null )  {^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P6_Replace_Expression]^if  ( actual == null )  {^20^^^^^19^24^if  ( this.wanted == null )  {^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P8_Replace_Mix]^if  ( this.wanted == false )  {^20^^^^^19^24^if  ( this.wanted == null )  {^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P9_Replace_Statement]^if  ( object == null )  {^20^^^^^19^24^if  ( this.wanted == null )  {^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P15_Unwrap_Block]^return actual == null;^20^21^22^^^19^24^if  ( this.wanted == null )  { return actual == null; }^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P16_Remove_Block]^^20^21^22^^^19^24^if  ( this.wanted == null )  { return actual == null; }^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P2_Replace_Operator]^return actual != null;^21^^^^^19^24^return actual == null;^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P5_Replace_Variable]^return wanted == null;^21^^^^^19^24^return actual == null;^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P8_Replace_Mix]^return actual != null;;^21^^^^^19^24^return actual == null;^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P8_Replace_Mix]^return wanted != null;;^21^^^^^19^24^return actual == null;^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P5_Replace_Variable]^return actual.equals ( wanted ) ;^23^^^^^19^24^return wanted.equals ( actual ) ;^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P7_Replace_Invocation]^return wanted .getClass (  )  ;^23^^^^^19^24^return wanted.equals ( actual ) ;^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P14_Delete_Statement]^^23^^^^^19^24^return wanted.equals ( actual ) ;^[CLASS] Equals 1  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] Object  actual  wanted  boolean  
[P5_Replace_Variable]^description.appendText ( describe ( actual )  ) ;^27^^^^^26^28^description.appendText ( describe ( wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P7_Replace_Invocation]^description.appendText ( quoting ( wanted )  ) ;^27^^^^^26^28^description.appendText ( describe ( wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P8_Replace_Mix]^description.appendText ( quoting ( actual )  ) ;^27^^^^^26^28^description.appendText ( describe ( wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P14_Delete_Statement]^^27^^^^^26^28^description.appendText ( describe ( wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  wanted  Description  description  boolean  
[P1_Replace_Type]^char text = quoting (  ) ;^31^^^^^30^39^String text = quoting (  ) ;^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P14_Delete_Statement]^^31^^^^^30^39^String text = quoting (  ) ;^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P11_Insert_Donor_Statement]^text+= quoting (  ) ;String text = quoting (  ) ;^31^^^^^30^39^String text = quoting (  ) ;^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P2_Replace_Operator]^if  ( object != null )  {^32^^^^^30^39^if  ( object == null )  {^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P5_Replace_Variable]^if  ( wanted == null )  {^32^^^^^30^39^if  ( object == null )  {^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P8_Replace_Mix]^if  ( object == this )  {^32^^^^^30^39^if  ( object == null )  {^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P9_Replace_Statement]^if  ( this.wanted == null )  {^32^^^^^30^39^if  ( object == null )  {^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P15_Unwrap_Block]^text += "null";^32^33^34^35^36^30^39^if  ( object == null )  { text+="null"; } else { text+=object.toString (  ) ; }^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P16_Remove_Block]^^32^33^34^35^36^30^39^if  ( object == null )  { text+="null"; } else { text+=object.toString (  ) ; }^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P1_Replace_Type]^text+=object.tochar (  ) ;^35^^^^^30^39^text+=object.toString (  ) ;^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P7_Replace_Invocation]^text+=object.equals (  ) ;^35^^^^^30^39^text+=object.toString (  ) ;^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P8_Replace_Mix]^text+= wanted.toString (  ) ;^35^^^^^30^39^text+=object.toString (  ) ;^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P11_Insert_Donor_Statement]^text+= quoting (  ) ;text+=object.toString (  ) ;^35^^^^^30^39^text+=object.toString (  ) ;^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P14_Delete_Statement]^^35^^^^^30^39^text+=object.toString (  ) ;^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P3_Replace_Literal]^text+="ull";^33^^^^^30^39^text+="null";^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P8_Replace_Mix]^text+="true";^33^^^^^30^39^text+="null";^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P3_Replace_Literal]^text+="";^33^^^^^30^39^text+="null";^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P8_Replace_Mix]^text+ =  text+;^37^^^^^30^39^text+= quoting (  ) ;^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P11_Insert_Donor_Statement]^text+=object.toString (  ) ;text+= quoting (  ) ;^37^^^^^30^39^text+= quoting (  ) ;^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P14_Delete_Statement]^^37^38^^^^30^39^text+= quoting (  ) ; return text;^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P11_Insert_Donor_Statement]^String text = quoting (  ) ;text+= quoting (  ) ;^37^^^^^30^39^text+= quoting (  ) ;^[CLASS] Equals 1  [METHOD] describe [RETURN_TYPE] String   Object object [VARIABLES] Object  actual  object  wanted  String  text  boolean  
[P2_Replace_Operator]^if  ( wanted  &  String )  {^42^^^^^41^49^if  ( wanted instanceof String )  {^[CLASS] Equals 1  [METHOD] quoting [RETURN_TYPE] String   [VARIABLES] Object  actual  object  wanted  boolean  
[P2_Replace_Operator]^if  ( wanted  !=  String )  {^42^^^^^41^49^if  ( wanted instanceof String )  {^[CLASS] Equals 1  [METHOD] quoting [RETURN_TYPE] String   [VARIABLES] Object  actual  object  wanted  boolean  
[P5_Replace_Variable]^if  ( object instanceof String )  {^42^^^^^41^49^if  ( wanted instanceof String )  {^[CLASS] Equals 1  [METHOD] quoting [RETURN_TYPE] String   [VARIABLES] Object  actual  object  wanted  boolean  
[P6_Replace_Expression]^if  ( wanted instanceof Character )  {^42^^^^^41^49^if  ( wanted instanceof String )  {^[CLASS] Equals 1  [METHOD] quoting [RETURN_TYPE] String   [VARIABLES] Object  actual  object  wanted  boolean  
[P2_Replace_Operator]^} else if  ( wanted  ==  Character )  {^44^^^^^41^49^} else if  ( wanted instanceof Character )  {^[CLASS] Equals 1  [METHOD] quoting [RETURN_TYPE] String   [VARIABLES] Object  actual  object  wanted  boolean  
[P5_Replace_Variable]^} else if  ( object instanceof Character )  {^44^^^^^41^49^} else if  ( wanted instanceof Character )  {^[CLASS] Equals 1  [METHOD] quoting [RETURN_TYPE] String   [VARIABLES] Object  actual  object  wanted  boolean  
[P6_Replace_Expression]^} else {^44^^^^^41^49^} else if  ( wanted instanceof Character )  {^[CLASS] Equals 1  [METHOD] quoting [RETURN_TYPE] String   [VARIABLES] Object  actual  object  wanted  boolean  
[P15_Unwrap_Block]^return "'";^44^45^46^47^48^41^49^} else if  ( wanted instanceof Character )  { return "'"; } else { return ""; }^[CLASS] Equals 1  [METHOD] quoting [RETURN_TYPE] String   [VARIABLES] Object  actual  object  wanted  boolean  
[P16_Remove_Block]^^44^45^46^47^48^41^49^} else if  ( wanted instanceof Character )  { return "'"; } else { return ""; }^[CLASS] Equals 1  [METHOD] quoting [RETURN_TYPE] String   [VARIABLES] Object  actual  object  wanted  boolean  
[P3_Replace_Literal]^return "";^45^^^^^41^49^return "'";^[CLASS] Equals 1  [METHOD] quoting [RETURN_TYPE] String   [VARIABLES] Object  actual  object  wanted  boolean  
[P2_Replace_Operator]^} else if  ( wanted  >=  Character )  {^44^^^^^41^49^} else if  ( wanted instanceof Character )  {^[CLASS] Equals 1  [METHOD] quoting [RETURN_TYPE] String   [VARIABLES] Object  actual  object  wanted  boolean  
[P3_Replace_Literal]^return "\"\";^43^^^^^41^49^return "\"";^[CLASS] Equals 1  [METHOD] quoting [RETURN_TYPE] String   [VARIABLES] Object  actual  object  wanted  boolean  
[P5_Replace_Variable]^return object;^52^^^^^51^53^return wanted;^[CLASS] Equals 1  [METHOD] getWanted [RETURN_TYPE] Object   [VARIABLES] Object  actual  object  wanted  boolean  
[P2_Replace_Operator]^if  ( o == null && !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^57^^^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P2_Replace_Operator]^if  ( o != null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^57^^^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P5_Replace_Variable]^if  ( wanted == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^57^^^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P6_Replace_Expression]^if  ( o == null ) {^57^^^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P6_Replace_Expression]^if  (  !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^57^^^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P7_Replace_Invocation]^if  ( o == null || !this.getClass (  )  .getClass (  )   )  {^57^^^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P7_Replace_Invocation]^if  ( o == null || !this.toString (  ) .equals ( o.getClass (  )  )  )  {^57^^^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P7_Replace_Invocation]^if  ( o == null || !this.equals (  ) .equals ( o.getClass (  )  )  )  {^57^^^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P8_Replace_Mix]^if  ( o == this ) {^57^^^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P15_Unwrap_Block]^return false;^57^58^59^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  { return false; }^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P16_Remove_Block]^^57^58^59^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  { return false; }^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P3_Replace_Literal]^return true;^58^^^^^56^62^return false;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P7_Replace_Invocation]^if  ( o == null || !this .equals ( wanted )  .equals ( o^57^^^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P7_Replace_Invocation]^if  ( o == null || !this .equals ( object )  .equals ( o^57^^^^^56^62^if  ( o == null || !this.getClass (  ) .equals ( o.getClass (  )  )  )  {^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P2_Replace_Operator]^return this.wanted == null && other.wanted == null && this.wanted != null && this.wanted.equals ( other.wanted ) ;^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P2_Replace_Operator]^return this.wanted == null || other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P2_Replace_Operator]^return this.wanted != null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P2_Replace_Operator]^return this.wanted == null && other.wanted != null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P2_Replace_Operator]^return this.wanted == null && other.wanted == null || this.wanted == null && this.wanted.equals ( other.wanted ) ;^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P5_Replace_Variable]^return wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P5_Replace_Variable]^return this.wanted == null && wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P5_Replace_Variable]^return other == null && this.wanted.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P5_Replace_Variable]^return other.wanted == null && this.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P7_Replace_Invocation]^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted .getClass (  )  ;^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P8_Replace_Mix]^return this.wanted == this && wanted == this || this.wanted != this && this.wanted .getClass (  )  ;^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P5_Replace_Variable]^return this.wanted == null && other.wanted.wanted == null || this.wanted != null && this.wanted.equals ( other ) ;^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P14_Delete_Statement]^^61^^^^^56^62^return this.wanted == null && other.wanted == null || this.wanted != null && this.wanted.equals ( other.wanted ) ;^[CLASS] Equals 1  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  actual  o  object  wanted  Equals  other  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  <<  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^69^74^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] withExtraTypeInfo [RETURN_TYPE] SelfDescribing   [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  <  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^69^74^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] withExtraTypeInfo [RETURN_TYPE] SelfDescribing   [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  !=  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^69^74^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] withExtraTypeInfo [RETURN_TYPE] SelfDescribing   [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P5_Replace_Variable]^description.appendText ( describe ( " ( "+ object.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^69^74^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] withExtraTypeInfo [RETURN_TYPE] SelfDescribing   [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P7_Replace_Invocation]^description.appendText ( describe ( " ( "+ wanted.toString (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^69^74^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] withExtraTypeInfo [RETURN_TYPE] SelfDescribing   [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P8_Replace_Mix]^description.appendText ( describe ( " ( "+ object .equals ( o )  .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^69^74^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] withExtraTypeInfo [RETURN_TYPE] SelfDescribing   [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P14_Delete_Statement]^^72^^^^^69^74^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] withExtraTypeInfo [RETURN_TYPE] SelfDescribing   [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  &  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^69^74^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] withExtraTypeInfo [RETURN_TYPE] SelfDescribing   [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  &&  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^69^74^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] withExtraTypeInfo [RETURN_TYPE] SelfDescribing   [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  ^  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^69^74^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] withExtraTypeInfo [RETURN_TYPE] SelfDescribing   [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P14_Delete_Statement]^^72^73^^^^69^74^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ; }};^[CLASS] Equals 1  [METHOD] withExtraTypeInfo [RETURN_TYPE] SelfDescribing   [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  >=  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (   instanceof   ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  ^  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P5_Replace_Variable]^description.appendText ( describe ( " ( "+ object.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P7_Replace_Invocation]^description.appendText ( describe ( " ( "+ wanted.toString (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P8_Replace_Mix]^description.appendText ( describe ( " ( "+ object .equals ( actual )  .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P14_Delete_Statement]^^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  >  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  !=  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  <=  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P14_Delete_Statement]^^72^73^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ; }};^[CLASS] Equals 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Object  actual  o  object  wanted  Description  description  boolean  
[P2_Replace_Operator]^return wanted != null || object != null && object.getClass (  )  == wanted.getClass (  ) ;^77^^^^^76^78^return wanted != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^[CLASS] Equals 1  [METHOD] typeMatches [RETURN_TYPE] boolean   Object object [VARIABLES] Object  actual  o  object  wanted  boolean  
[P2_Replace_Operator]^return wanted == null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^77^^^^^76^78^return wanted != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^[CLASS] Equals 1  [METHOD] typeMatches [RETURN_TYPE] boolean   Object object [VARIABLES] Object  actual  o  object  wanted  boolean  
[P2_Replace_Operator]^return wanted != null && object == null && object.getClass (  )  == wanted.getClass (  ) ;^77^^^^^76^78^return wanted != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^[CLASS] Equals 1  [METHOD] typeMatches [RETURN_TYPE] boolean   Object object [VARIABLES] Object  actual  o  object  wanted  boolean  
[P2_Replace_Operator]^return wanted != null && object != null && object.getClass (  )  != wanted.getClass (  ) ;^77^^^^^76^78^return wanted != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^[CLASS] Equals 1  [METHOD] typeMatches [RETURN_TYPE] boolean   Object object [VARIABLES] Object  actual  o  object  wanted  boolean  
[P5_Replace_Variable]^return o != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^77^^^^^76^78^return wanted != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^[CLASS] Equals 1  [METHOD] typeMatches [RETURN_TYPE] boolean   Object object [VARIABLES] Object  actual  o  object  wanted  boolean  
[P5_Replace_Variable]^return object != null && wanted != null && object.getClass (  )  == wanted.getClass (  ) ;^77^^^^^76^78^return wanted != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^[CLASS] Equals 1  [METHOD] typeMatches [RETURN_TYPE] boolean   Object object [VARIABLES] Object  actual  o  object  wanted  boolean  
[P7_Replace_Invocation]^return wanted != null && object != null && object.equals (  )  == wanted.getClass (  ) ;^77^^^^^76^78^return wanted != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^[CLASS] Equals 1  [METHOD] typeMatches [RETURN_TYPE] boolean   Object object [VARIABLES] Object  actual  o  object  wanted  boolean  
[P7_Replace_Invocation]^return wanted != null && object != null && object.toString (  )  == wanted.getClass (  ) ;^77^^^^^76^78^return wanted != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^[CLASS] Equals 1  [METHOD] typeMatches [RETURN_TYPE] boolean   Object object [VARIABLES] Object  actual  o  object  wanted  boolean  
[P7_Replace_Invocation]^return wanted != null && object != null && object .equals ( object )   == wanted^77^^^^^76^78^return wanted != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^[CLASS] Equals 1  [METHOD] typeMatches [RETURN_TYPE] boolean   Object object [VARIABLES] Object  actual  o  object  wanted  boolean  
[P5_Replace_Variable]^return wanted != null && wanted != null && object.getClass (  )  == wanted.getClass (  ) ;^77^^^^^76^78^return wanted != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^[CLASS] Equals 1  [METHOD] typeMatches [RETURN_TYPE] boolean   Object object [VARIABLES] Object  actual  o  object  wanted  boolean  
[P14_Delete_Statement]^^77^^^^^76^78^return wanted != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^[CLASS] Equals 1  [METHOD] typeMatches [RETURN_TYPE] boolean   Object object [VARIABLES] Object  actual  o  object  wanted  boolean  
[P5_Replace_Variable]^return object != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^77^^^^^76^78^return wanted != null && object != null && object.getClass (  )  == wanted.getClass (  ) ;^[CLASS] Equals 1  [METHOD] typeMatches [RETURN_TYPE] boolean   Object object [VARIABLES] Object  actual  o  object  wanted  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  ==  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  <=  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (   instanceof   ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Description  description  boolean  
[P14_Delete_Statement]^^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Description  description  boolean  
[P2_Replace_Operator]^description.appendText ( describe ( " ( "+ wanted.getClass (  >>  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^72^^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ;^[CLASS] 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Description  description  boolean  
[P14_Delete_Statement]^^72^73^^^^71^73^description.appendText ( describe ( " ( "+ wanted.getClass (  ) .getSimpleName (  )  +" )  " + wanted )  ) ; }};^[CLASS] 1  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Description  description  boolean  
