[P1_Replace_Type]^public  long  size (  )  { return 0; }^74^^^^^69^79^public int size (  )  { return 0; }^[CLASS] JsonNode  [METHOD] size [RETURN_TYPE] int   [VARIABLES] boolean  
[P3_Replace_Literal]^public int size (  )  { return 9; }^74^^^^^69^79^public int size (  )  { return 0; }^[CLASS] JsonNode  [METHOD] size [RETURN_TYPE] int   [VARIABLES] boolean  
[P3_Replace_Literal]^public int size() - 5  { return 0; }^74^^^^^69^79^public int size (  )  { return 0; }^[CLASS] JsonNode  [METHOD] size [RETURN_TYPE] int   [VARIABLES] boolean  
[P8_Replace_Mix]^public int size (  )  { return 0 ; }^74^^^^^69^79^public int size (  )  { return 0; }^[CLASS] JsonNode  [METHOD] size [RETURN_TYPE] int   [VARIABLES] boolean  
[P3_Replace_Literal]^return true;^81^^^^^77^85^return false;^[CLASS] JsonNode  [METHOD] isValueNode [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P3_Replace_Literal]^return false;^83^^^^^77^85^return true;^[CLASS] JsonNode  [METHOD] isValueNode [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P7_Replace_Invocation]^switch  ( elements (  )  )  {^79^^^^^77^85^switch  ( getNodeType (  )  )  {^[CLASS] JsonNode  [METHOD] isValueNode [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P14_Delete_Statement]^^79^80^81^^^77^85^switch  ( getNodeType (  )  )  { case ARRAY: case OBJECT: case MISSING: return false;^[CLASS] JsonNode  [METHOD] isValueNode [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P7_Replace_Invocation]^final JsonNodeType type = elements (  ) ;^89^^^^^88^91^final JsonNodeType type = getNodeType (  ) ;^[CLASS] JsonNode  [METHOD] isContainerNode [RETURN_TYPE] boolean   [VARIABLES] boolean  JsonNodeType  type  
[P14_Delete_Statement]^^89^90^^^^88^91^final JsonNodeType type = getNodeType (  ) ; return type == JsonNodeType.OBJECT || type == JsonNodeType.ARRAY;^[CLASS] JsonNode  [METHOD] isContainerNode [RETURN_TYPE] boolean   [VARIABLES] boolean  JsonNodeType  type  
[P2_Replace_Operator]^return type == JsonNodeType.OBJECT && type == JsonNodeType.ARRAY;^90^^^^^88^91^return type == JsonNodeType.OBJECT || type == JsonNodeType.ARRAY;^[CLASS] JsonNode  [METHOD] isContainerNode [RETURN_TYPE] boolean   [VARIABLES] boolean  JsonNodeType  type  
[P2_Replace_Operator]^return type != JsonNodeType.OBJECT || type == JsonNodeType.ARRAY;^90^^^^^88^91^return type == JsonNodeType.OBJECT || type == JsonNodeType.ARRAY;^[CLASS] JsonNode  [METHOD] isContainerNode [RETURN_TYPE] boolean   [VARIABLES] boolean  JsonNodeType  type  
[P8_Replace_Mix]^return type ;^90^^^^^88^91^return type == JsonNodeType.OBJECT || type == JsonNodeType.ARRAY;^[CLASS] JsonNode  [METHOD] isContainerNode [RETURN_TYPE] boolean   [VARIABLES] boolean  JsonNodeType  type  
[P2_Replace_Operator]^return getNodeType (  )  >= JsonNodeType.MISSING;^95^^^^^94^96^return getNodeType (  )  == JsonNodeType.MISSING;^[CLASS] JsonNode  [METHOD] isMissingNode [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P7_Replace_Invocation]^return elements (  )  == JsonNodeType.MISSING;^95^^^^^94^96^return getNodeType (  )  == JsonNodeType.MISSING;^[CLASS] JsonNode  [METHOD] isMissingNode [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P14_Delete_Statement]^^95^^^^^94^96^return getNodeType (  )  == JsonNodeType.MISSING;^[CLASS] JsonNode  [METHOD] isMissingNode [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P2_Replace_Operator]^return getNodeType (  )  <= JsonNodeType.ARRAY;^100^^^^^99^101^return getNodeType (  )  == JsonNodeType.ARRAY;^[CLASS] JsonNode  [METHOD] isArray [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P7_Replace_Invocation]^return elements (  )  == JsonNodeType.ARRAY;^100^^^^^99^101^return getNodeType (  )  == JsonNodeType.ARRAY;^[CLASS] JsonNode  [METHOD] isArray [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P14_Delete_Statement]^^100^^^^^99^101^return getNodeType (  )  == JsonNodeType.ARRAY;^[CLASS] JsonNode  [METHOD] isArray [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P2_Replace_Operator]^return getNodeType (  )  < JsonNodeType.OBJECT;^105^^^^^104^106^return getNodeType (  )  == JsonNodeType.OBJECT;^[CLASS] JsonNode  [METHOD] isObject [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P7_Replace_Invocation]^return elements (  )  == JsonNodeType.OBJECT;^105^^^^^104^106^return getNodeType (  )  == JsonNodeType.OBJECT;^[CLASS] JsonNode  [METHOD] isObject [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P14_Delete_Statement]^^105^^^^^104^106^return getNodeType (  )  == JsonNodeType.OBJECT;^[CLASS] JsonNode  [METHOD] isObject [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P8_Replace_Mix]^public JsonNode get ( String fieldName )  { return this; }^148^^^^^143^153^public JsonNode get ( String fieldName )  { return null; }^[CLASS] JsonNode  [METHOD] get [RETURN_TYPE] JsonNode   String fieldName [VARIABLES] boolean  String  fieldName  
[P14_Delete_Statement]^^175^^^^^174^176^return EmptyIterator.instance (  ) ;^[CLASS] JsonNode  [METHOD] fieldNames [RETURN_TYPE] Iterator   [VARIABLES] boolean  
[P2_Replace_Operator]^return getNodeType (  )  > JsonNodeType.POJO;^204^^^^^203^205^return getNodeType (  )  == JsonNodeType.POJO;^[CLASS] JsonNode  [METHOD] isPojo [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P7_Replace_Invocation]^return elements (  )  == JsonNodeType.POJO;^204^^^^^203^205^return getNodeType (  )  == JsonNodeType.POJO;^[CLASS] JsonNode  [METHOD] isPojo [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P14_Delete_Statement]^^204^^^^^203^205^return getNodeType (  )  == JsonNodeType.POJO;^[CLASS] JsonNode  [METHOD] isPojo [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P2_Replace_Operator]^return getNodeType (  )  < JsonNodeType.NUMBER;^211^^^^^210^212^return getNodeType (  )  == JsonNodeType.NUMBER;^[CLASS] JsonNode  [METHOD] isNumber [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P7_Replace_Invocation]^return elements (  )  == JsonNodeType.NUMBER;^211^^^^^210^212^return getNodeType (  )  == JsonNodeType.NUMBER;^[CLASS] JsonNode  [METHOD] isNumber [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P14_Delete_Statement]^^211^^^^^210^212^return getNodeType (  )  == JsonNodeType.NUMBER;^[CLASS] JsonNode  [METHOD] isNumber [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P3_Replace_Literal]^public boolean isIntegralNumber (  )  { return true; }^219^^^^^214^224^public boolean isIntegralNumber (  )  { return false; }^[CLASS] JsonNode  [METHOD] isIntegralNumber [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P3_Replace_Literal]^public boolean isFloatingPointNumber (  )  { return true; }^225^^^^^220^230^public boolean isFloatingPointNumber (  )  { return false; }^[CLASS] JsonNode  [METHOD] isFloatingPointNumber [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P3_Replace_Literal]^public boolean isShort (  )  { return true; }^237^^^^^232^242^public boolean isShort (  )  { return false; }^[CLASS] JsonNode  [METHOD] isShort [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P3_Replace_Literal]^public boolean isInt (  )  { return true; }^249^^^^^244^254^public boolean isInt (  )  { return false; }^[CLASS] JsonNode  [METHOD] isInt [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P3_Replace_Literal]^public boolean isLong (  )  { return true; }^261^^^^^256^266^public boolean isLong (  )  { return false; }^[CLASS] JsonNode  [METHOD] isLong [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P3_Replace_Literal]^public boolean isFloat (  )  { return true; }^266^^^^^261^271^public boolean isFloat (  )  { return false; }^[CLASS] JsonNode  [METHOD] isFloat [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P3_Replace_Literal]^public boolean isDouble (  )  { return true; }^268^^^^^263^273^public boolean isDouble (  )  { return false; }^[CLASS] JsonNode  [METHOD] isDouble [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P3_Replace_Literal]^public boolean isBigDecimal (  )  { return true; }^269^^^^^264^274^public boolean isBigDecimal (  )  { return false; }^[CLASS] JsonNode  [METHOD] isBigDecimal [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P3_Replace_Literal]^public boolean isBigInteger (  )  { return true; }^270^^^^^265^275^public boolean isBigInteger (  )  { return false; }^[CLASS] JsonNode  [METHOD] isBigInteger [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P2_Replace_Operator]^return getNodeType (  )  != JsonNodeType.STRING;^277^^^^^276^278^return getNodeType (  )  == JsonNodeType.STRING;^[CLASS] JsonNode  [METHOD] isTextual [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P7_Replace_Invocation]^return elements (  )  == JsonNodeType.STRING;^277^^^^^276^278^return getNodeType (  )  == JsonNodeType.STRING;^[CLASS] JsonNode  [METHOD] isTextual [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P14_Delete_Statement]^^277^^^^^276^278^return getNodeType (  )  == JsonNodeType.STRING;^[CLASS] JsonNode  [METHOD] isTextual [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P2_Replace_Operator]^return getNodeType (  )  != JsonNodeType.BOOLEAN;^285^^^^^284^286^return getNodeType (  )  == JsonNodeType.BOOLEAN;^[CLASS] JsonNode  [METHOD] isBoolean [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P7_Replace_Invocation]^return elements (  )  == JsonNodeType.BOOLEAN;^285^^^^^284^286^return getNodeType (  )  == JsonNodeType.BOOLEAN;^[CLASS] JsonNode  [METHOD] isBoolean [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P14_Delete_Statement]^^285^^^^^284^286^return getNodeType (  )  == JsonNodeType.BOOLEAN;^[CLASS] JsonNode  [METHOD] isBoolean [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P2_Replace_Operator]^return getNodeType (  )  >= JsonNodeType.NULL;^293^^^^^292^294^return getNodeType (  )  == JsonNodeType.NULL;^[CLASS] JsonNode  [METHOD] isNull [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P7_Replace_Invocation]^return elements (  )  == JsonNodeType.NULL;^293^^^^^292^294^return getNodeType (  )  == JsonNodeType.NULL;^[CLASS] JsonNode  [METHOD] isNull [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P8_Replace_Mix]^return elements (  )   ||  JsonNodeType.NULL;^293^^^^^292^294^return getNodeType (  )  == JsonNodeType.NULL;^[CLASS] JsonNode  [METHOD] isNull [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P14_Delete_Statement]^^293^^^^^292^294^return getNodeType (  )  == JsonNodeType.NULL;^[CLASS] JsonNode  [METHOD] isNull [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P2_Replace_Operator]^return getNodeType (  )  != JsonNodeType.BINARY;^305^^^^^304^306^return getNodeType (  )  == JsonNodeType.BINARY;^[CLASS] JsonNode  [METHOD] isBinary [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P7_Replace_Invocation]^return elements (  )  == JsonNodeType.BINARY;^305^^^^^304^306^return getNodeType (  )  == JsonNodeType.BINARY;^[CLASS] JsonNode  [METHOD] isBinary [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P8_Replace_Mix]^return elements (  )   ||  JsonNodeType.BINARY;^305^^^^^304^306^return getNodeType (  )  == JsonNodeType.BINARY;^[CLASS] JsonNode  [METHOD] isBinary [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P14_Delete_Statement]^^305^^^^^304^306^return getNodeType (  )  == JsonNodeType.BINARY;^[CLASS] JsonNode  [METHOD] isBinary [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P3_Replace_Literal]^public boolean canConvertToInt (  )  { return true; }^317^^^^^312^322^public boolean canConvertToInt (  )  { return false; }^[CLASS] JsonNode  [METHOD] canConvertToInt [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P3_Replace_Literal]^public boolean canConvertToLong (  )  { return true; }^328^^^^^323^333^public boolean canConvertToLong (  )  { return false; }^[CLASS] JsonNode  [METHOD] canConvertToLong [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P8_Replace_Mix]^public String textValue (  )  { return this; }^346^^^^^341^351^public String textValue (  )  { return null; }^[CLASS] JsonNode  [METHOD] textValue [RETURN_TYPE] String   [VARIABLES] boolean  
[P8_Replace_Mix]^return this;^359^^^^^358^360^return null;^[CLASS] JsonNode  [METHOD] binaryValue [RETURN_TYPE] byte[]   [VARIABLES] boolean  
[P3_Replace_Literal]^public boolean booleanValue (  )  { return true; }^370^^^^^365^375^public boolean booleanValue (  )  { return false; }^[CLASS] JsonNode  [METHOD] booleanValue [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P8_Replace_Mix]^public Number numberValue (  )  { return true; }^380^^^^^375^385^public Number numberValue (  )  { return null; }^[CLASS] JsonNode  [METHOD] numberValue [RETURN_TYPE] Number   [VARIABLES] boolean  
[P3_Replace_Literal]^public short shortValue (  )  { return -4; }^392^^^^^387^397^public short shortValue (  )  { return 0; }^[CLASS] JsonNode  [METHOD] shortValue [RETURN_TYPE] short   [VARIABLES] boolean  
[P8_Replace_Mix]^public short shortValue (  )  { return 3; }^392^^^^^387^397^public short shortValue (  )  { return 0; }^[CLASS] JsonNode  [METHOD] shortValue [RETURN_TYPE] short   [VARIABLES] boolean  
[P1_Replace_Type]^public  long  intValue (  )  { return 0; }^404^^^^^399^409^public int intValue (  )  { return 0; }^[CLASS] JsonNode  [METHOD] intValue [RETURN_TYPE] int   [VARIABLES] boolean  
[P3_Replace_Literal]^public int intValue (  )  { return 1; }^404^^^^^399^409^public int intValue (  )  { return 0; }^[CLASS] JsonNode  [METHOD] intValue [RETURN_TYPE] int   [VARIABLES] boolean  
[P8_Replace_Mix]^public int intValue (  )  { return 0 / 1; }^404^^^^^399^409^public int intValue (  )  { return 0; }^[CLASS] JsonNode  [METHOD] intValue [RETURN_TYPE] int   [VARIABLES] boolean  
[P1_Replace_Type]^public  short  longValue (  )  { return 0L; }^416^^^^^411^421^public long longValue (  )  { return 0L; }^[CLASS] JsonNode  [METHOD] longValue [RETURN_TYPE] long   [VARIABLES] boolean  
[P8_Replace_Mix]^public long longValue (  )  { return 0; }^416^^^^^411^421^public long longValue (  )  { return 0L; }^[CLASS] JsonNode  [METHOD] longValue [RETURN_TYPE] long   [VARIABLES] boolean  
[P1_Replace_Type]^public int floatValue (  )  { return 0.0f; }^429^^^^^424^434^public float floatValue (  )  { return 0.0f; }^[CLASS] JsonNode  [METHOD] floatValue [RETURN_TYPE] float   [VARIABLES] boolean  
[P1_Replace_Type]^public int doubleValue (  )  { return 0.0; }^442^^^^^437^447^public double doubleValue (  )  { return 0.0; }^[CLASS] JsonNode  [METHOD] doubleValue [RETURN_TYPE] double   [VARIABLES] boolean  
[P8_Replace_Mix]^public double doubleValue (  )  { return 0.0d; }^442^^^^^437^447^public double doubleValue (  )  { return 0.0; }^[CLASS] JsonNode  [METHOD] doubleValue [RETURN_TYPE] double   [VARIABLES] boolean  
[P3_Replace_Literal]^return asInt ( 7 ) ;^472^^^^^471^473^return asInt ( 0 ) ;^[CLASS] JsonNode  [METHOD] asInt [RETURN_TYPE] int   [VARIABLES] boolean  
[P7_Replace_Invocation]^return get ( 0 ) ;^472^^^^^471^473^return asInt ( 0 ) ;^[CLASS] JsonNode  [METHOD] asInt [RETURN_TYPE] int   [VARIABLES] boolean  
[P8_Replace_Mix]^return get ( 0 - 3 ) ;^472^^^^^471^473^return asInt ( 0 ) ;^[CLASS] JsonNode  [METHOD] asInt [RETURN_TYPE] int   [VARIABLES] boolean  
[P3_Replace_Literal]^return asInt ( 2 ) ;^472^^^^^471^473^return asInt ( 0 ) ;^[CLASS] JsonNode  [METHOD] asInt [RETURN_TYPE] int   [VARIABLES] boolean  
[P14_Delete_Statement]^^472^^^^^471^473^return asInt ( 0 ) ;^[CLASS] JsonNode  [METHOD] asInt [RETURN_TYPE] int   [VARIABLES] boolean  
[P7_Replace_Invocation]^return asInt ( 0L ) ;^500^^^^^499^501^return asLong ( 0L ) ;^[CLASS] JsonNode  [METHOD] asLong [RETURN_TYPE] long   [VARIABLES] boolean  
[P14_Delete_Statement]^^500^^^^^499^501^return asLong ( 0L ) ;^[CLASS] JsonNode  [METHOD] asLong [RETURN_TYPE] long   [VARIABLES] boolean  
[P7_Replace_Invocation]^return asBoolean ( 0.0 ) ;^528^^^^^527^529^return asDouble ( 0.0 ) ;^[CLASS] JsonNode  [METHOD] asDouble [RETURN_TYPE] double   [VARIABLES] boolean  
[P14_Delete_Statement]^^528^^^^^527^529^return asDouble ( 0.0 ) ;^[CLASS] JsonNode  [METHOD] asDouble [RETURN_TYPE] double   [VARIABLES] boolean  
[P3_Replace_Literal]^return asBoolean ( true ) ;^556^^^^^555^557^return asBoolean ( false ) ;^[CLASS] JsonNode  [METHOD] asBoolean [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P7_Replace_Invocation]^return asDouble ( false ) ;^556^^^^^555^557^return asBoolean ( false ) ;^[CLASS] JsonNode  [METHOD] asBoolean [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P14_Delete_Statement]^^556^^^^^555^557^return asBoolean ( false ) ;^[CLASS] JsonNode  [METHOD] asBoolean [RETURN_TYPE] boolean   [VARIABLES] boolean  
[P2_Replace_Operator]^return get ( fieldName )  == null;^600^^^^^599^601^return get ( fieldName )  != null;^[CLASS] JsonNode  [METHOD] has [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  String  fieldName  
[P7_Replace_Invocation]^return has ( fieldName )  != null;^600^^^^^599^601^return get ( fieldName )  != null;^[CLASS] JsonNode  [METHOD] has [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  String  fieldName  
[P8_Replace_Mix]^return has ( fieldName )  != true;^600^^^^^599^601^return get ( fieldName )  != null;^[CLASS] JsonNode  [METHOD] has [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  String  fieldName  
[P14_Delete_Statement]^^600^^^^^599^601^return get ( fieldName )  != null;^[CLASS] JsonNode  [METHOD] has [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  String  fieldName  
[P2_Replace_Operator]^return get ( index )  == null;^626^^^^^625^627^return get ( index )  != null;^[CLASS] JsonNode  [METHOD] has [RETURN_TYPE] boolean   int index [VARIABLES] boolean  int  index  
[P7_Replace_Invocation]^return has ( index )  != null;^626^^^^^625^627^return get ( index )  != null;^[CLASS] JsonNode  [METHOD] has [RETURN_TYPE] boolean   int index [VARIABLES] boolean  int  index  
[P14_Delete_Statement]^^626^^^^^625^627^return get ( index )  != null;^[CLASS] JsonNode  [METHOD] has [RETURN_TYPE] boolean   int index [VARIABLES] boolean  int  index  
[P7_Replace_Invocation]^JsonNode n = has ( fieldName ) ;^641^^^^^640^643^JsonNode n = get ( fieldName ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  JsonNode  n  String  fieldName  
[P11_Insert_Donor_Statement]^JsonNode n = get ( index ) ;JsonNode n = get ( fieldName ) ;^641^^^^^640^643^JsonNode n = get ( fieldName ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  JsonNode  n  String  fieldName  
[P11_Insert_Donor_Statement]^List<JsonNode> result = findParents ( fieldName, null ) ;JsonNode n = get ( fieldName ) ;^641^^^^^640^643^JsonNode n = get ( fieldName ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  JsonNode  n  String  fieldName  
[P11_Insert_Donor_Statement]^List<JsonNode> result = findValues ( fieldName, null ) ;JsonNode n = get ( fieldName ) ;^641^^^^^640^643^JsonNode n = get ( fieldName ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  JsonNode  n  String  fieldName  
[P14_Delete_Statement]^^641^642^^^^640^643^JsonNode n = get ( fieldName ) ; return  ( n != null )  && !n.isNull (  ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  JsonNode  n  String  fieldName  
[P2_Replace_Operator]^return  ( n != null )  || !n.isNull (  ) ;^642^^^^^640^643^return  ( n != null )  && !n.isNull (  ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  JsonNode  n  String  fieldName  
[P2_Replace_Operator]^return  ( n == null )  && !n.isNull (  ) ;^642^^^^^640^643^return  ( n != null )  && !n.isNull (  ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  JsonNode  n  String  fieldName  
[P7_Replace_Invocation]^return  ( n != null )  && !n.asInt (  ) ;^642^^^^^640^643^return  ( n != null )  && !n.isNull (  ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  JsonNode  n  String  fieldName  
[P7_Replace_Invocation]^return  ( n != null )  && !n .hasNonNull ( fieldName )  ;^642^^^^^640^643^return  ( n != null )  && !n.isNull (  ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  JsonNode  n  String  fieldName  
[P14_Delete_Statement]^^642^^^^^640^643^return  ( n != null )  && !n.isNull (  ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   String fieldName [VARIABLES] boolean  JsonNode  n  String  fieldName  
[P7_Replace_Invocation]^JsonNode n = has ( index ) ;^657^^^^^656^659^JsonNode n = get ( index ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   int index [VARIABLES] boolean  JsonNode  n  int  index  
[P11_Insert_Donor_Statement]^JsonNode n = get ( fieldName ) ;JsonNode n = get ( index ) ;^657^^^^^656^659^JsonNode n = get ( index ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   int index [VARIABLES] boolean  JsonNode  n  int  index  
[P14_Delete_Statement]^^657^^^^^656^659^JsonNode n = get ( index ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   int index [VARIABLES] boolean  JsonNode  n  int  index  
[P2_Replace_Operator]^return  ( n != null )  || !n.isNull (  ) ;^658^^^^^656^659^return  ( n != null )  && !n.isNull (  ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   int index [VARIABLES] boolean  JsonNode  n  int  index  
[P2_Replace_Operator]^return  ( n == null )  && !n.isNull (  ) ;^658^^^^^656^659^return  ( n != null )  && !n.isNull (  ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   int index [VARIABLES] boolean  JsonNode  n  int  index  
[P7_Replace_Invocation]^return  ( n != null )  && !n.asInt (  ) ;^658^^^^^656^659^return  ( n != null )  && !n.isNull (  ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   int index [VARIABLES] boolean  JsonNode  n  int  index  
[P14_Delete_Statement]^^658^^^^^656^659^return  ( n != null )  && !n.isNull (  ) ;^[CLASS] JsonNode  [METHOD] hasNonNull [RETURN_TYPE] boolean   int index [VARIABLES] boolean  JsonNode  n  int  index  
[P7_Replace_Invocation]^public final Iterator<JsonNode> iterator (  )  { return isNull (  ) ; }^673^^^^^668^678^public final Iterator<JsonNode> iterator (  )  { return elements (  ) ; }^[CLASS] JsonNode  [METHOD] iterator [RETURN_TYPE] Iterator   [VARIABLES] boolean  
[P14_Delete_Statement]^^682^^^^^681^683^return EmptyIterator.instance (  ) ;^[CLASS] JsonNode  [METHOD] elements [RETURN_TYPE] Iterator   [VARIABLES] boolean  
[P14_Delete_Statement]^^690^^^^^689^691^return EmptyIterator.instance (  ) ;^[CLASS] JsonNode  [METHOD] fields [RETURN_TYPE] Iterator   [VARIABLES] boolean  
[P7_Replace_Invocation]^List<JsonNode> result = findParents ( fieldName, null ) ;^721^^^^^719^726^List<JsonNode> result = findValues ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findValues [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P11_Insert_Donor_Statement]^List<String> result = findValuesAsText ( fieldName, null ) ;List<JsonNode> result = findValues ( fieldName, null ) ;^721^^^^^719^726^List<JsonNode> result = findValues ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findValues [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P11_Insert_Donor_Statement]^List<JsonNode> result = findParents ( fieldName, null ) ;List<JsonNode> result = findValues ( fieldName, null ) ;^721^^^^^719^726^List<JsonNode> result = findValues ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findValues [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P11_Insert_Donor_Statement]^JsonNode n = get ( fieldName ) ;List<JsonNode> result = findValues ( fieldName, null ) ;^721^^^^^719^726^List<JsonNode> result = findValues ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findValues [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P14_Delete_Statement]^^721^^^^^719^726^List<JsonNode> result = findValues ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findValues [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P2_Replace_Operator]^if  ( result != null )  {^722^^^^^719^726^if  ( result == null )  {^[CLASS] JsonNode  [METHOD] findValues [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P8_Replace_Mix]^if  ( result == this )  {^722^^^^^719^726^if  ( result == null )  {^[CLASS] JsonNode  [METHOD] findValues [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P15_Unwrap_Block]^return java.util.Collections.emptyList();^722^723^724^^^719^726^if  ( result == null )  { return Collections.emptyList (  ) ; }^[CLASS] JsonNode  [METHOD] findValues [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P16_Remove_Block]^^722^723^724^^^719^726^if  ( result == null )  { return Collections.emptyList (  ) ; }^[CLASS] JsonNode  [METHOD] findValues [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P13_Insert_Block]^if  ( result == null )  {     return emptyList (  ) ; }^723^^^^^719^726^[Delete]^[CLASS] JsonNode  [METHOD] findValues [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P14_Delete_Statement]^^723^^^^^719^726^return Collections.emptyList (  ) ;^[CLASS] JsonNode  [METHOD] findValues [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P7_Replace_Invocation]^List<String> result = findParents ( fieldName, null ) ;^734^^^^^732^739^List<String> result = findValuesAsText ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findValuesAsText [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P11_Insert_Donor_Statement]^List<JsonNode> result = findParents ( fieldName, null ) ;List<String> result = findValuesAsText ( fieldName, null ) ;^734^^^^^732^739^List<String> result = findValuesAsText ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findValuesAsText [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P11_Insert_Donor_Statement]^List<JsonNode> result = findValues ( fieldName, null ) ;List<String> result = findValuesAsText ( fieldName, null ) ;^734^^^^^732^739^List<String> result = findValuesAsText ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findValuesAsText [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P14_Delete_Statement]^^734^^^^^732^739^List<String> result = findValuesAsText ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findValuesAsText [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P2_Replace_Operator]^if  ( result != null )  {^735^^^^^732^739^if  ( result == null )  {^[CLASS] JsonNode  [METHOD] findValuesAsText [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P8_Replace_Mix]^if  ( result == this )  {^735^^^^^732^739^if  ( result == null )  {^[CLASS] JsonNode  [METHOD] findValuesAsText [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P15_Unwrap_Block]^return java.util.Collections.emptyList();^735^736^737^^^732^739^if  ( result == null )  { return Collections.emptyList (  ) ; }^[CLASS] JsonNode  [METHOD] findValuesAsText [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P16_Remove_Block]^^735^736^737^^^732^739^if  ( result == null )  { return Collections.emptyList (  ) ; }^[CLASS] JsonNode  [METHOD] findValuesAsText [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P13_Insert_Block]^if  ( result == null )  {     return emptyList (  ) ; }^736^^^^^732^739^[Delete]^[CLASS] JsonNode  [METHOD] findValuesAsText [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P14_Delete_Statement]^^736^^^^^732^739^return Collections.emptyList (  ) ;^[CLASS] JsonNode  [METHOD] findValuesAsText [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P5_Replace_Variable]^return null;^738^^^^^732^739^return result;^[CLASS] JsonNode  [METHOD] findValuesAsText [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P7_Replace_Invocation]^List<JsonNode> result = findValuesAsText ( fieldName, null ) ;^777^^^^^775^782^List<JsonNode> result = findParents ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findParents [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P11_Insert_Donor_Statement]^List<String> result = findValuesAsText ( fieldName, null ) ;List<JsonNode> result = findParents ( fieldName, null ) ;^777^^^^^775^782^List<JsonNode> result = findParents ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findParents [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P11_Insert_Donor_Statement]^List<JsonNode> result = findValues ( fieldName, null ) ;List<JsonNode> result = findParents ( fieldName, null ) ;^777^^^^^775^782^List<JsonNode> result = findParents ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findParents [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P11_Insert_Donor_Statement]^JsonNode n = get ( fieldName ) ;List<JsonNode> result = findParents ( fieldName, null ) ;^777^^^^^775^782^List<JsonNode> result = findParents ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findParents [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P14_Delete_Statement]^^777^^^^^775^782^List<JsonNode> result = findParents ( fieldName, null ) ;^[CLASS] JsonNode  [METHOD] findParents [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P2_Replace_Operator]^if  ( result != null )  {^778^^^^^775^782^if  ( result == null )  {^[CLASS] JsonNode  [METHOD] findParents [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P8_Replace_Mix]^if  ( result == this )  {^778^^^^^775^782^if  ( result == null )  {^[CLASS] JsonNode  [METHOD] findParents [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P15_Unwrap_Block]^return java.util.Collections.emptyList();^778^779^780^^^775^782^if  ( result == null )  { return Collections.emptyList (  ) ; }^[CLASS] JsonNode  [METHOD] findParents [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P16_Remove_Block]^^778^779^780^^^775^782^if  ( result == null )  { return Collections.emptyList (  ) ; }^[CLASS] JsonNode  [METHOD] findParents [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P13_Insert_Block]^if  ( result == null )  {     return emptyList (  ) ; }^779^^^^^775^782^[Delete]^[CLASS] JsonNode  [METHOD] findParents [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P14_Delete_Statement]^^779^^^^^775^782^return Collections.emptyList (  ) ;^[CLASS] JsonNode  [METHOD] findParents [RETURN_TYPE] List   String fieldName [VARIABLES] boolean  List  result  String  fieldName  
[P2_Replace_Operator]^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ==  ( but " +getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;^803^804^^^^802^805^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] with [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P2_Replace_Operator]^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  >=  ( but " +getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;^803^804^^^^802^805^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] with [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P4_Replace_Constructor]^throw throw  new UnsupportedOperationException (  (  ( "JsonNode not of type ObjectNode  ( but " +  ( getClass (  ) .getName (  )  )  )  + " ) , can not call withArray (  )  on it" )  )  .getName (  ) +" ) , can not call with (  )  on it" ) ;^803^804^^^^802^805^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] with [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P2_Replace_Operator]^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  >>  ( but " +getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;^803^804^^^^802^805^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] with [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P11_Insert_Donor_Statement]^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;^803^804^^^^802^805^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] with [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P14_Delete_Statement]^^804^^^^^802^805^+getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] with [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P11_Insert_Donor_Statement]^+getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;+getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;^804^^^^^802^805^+getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] with [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P2_Replace_Operator]^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  !=  ( but " +getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^816^817^^^^815^818^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] withArray [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P2_Replace_Operator]^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  >=  ( but " +getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^816^817^^^^815^818^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] withArray [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P4_Replace_Constructor]^throw throw  new UnsupportedOperationException (  (  ( "JsonNode not of type ObjectNode  ( but " +  ( getClass (  ) .getName (  )  )  )  + " ) , can not call with (  )  on it" )  )  .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^816^817^^^^815^818^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] withArray [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P2_Replace_Operator]^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  &  ( but " +getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^816^817^^^^815^818^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] withArray [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P2_Replace_Operator]^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ^  ( but " +getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^816^817^^^^815^818^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] withArray [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P11_Insert_Donor_Statement]^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^816^817^^^^815^818^throw new UnsupportedOperationException ( "JsonNode not of type ObjectNode  ( but " +getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] withArray [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P14_Delete_Statement]^^817^^^^^815^818^+getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] withArray [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
[P11_Insert_Donor_Statement]^+getClass (  ) .getName (  ) +" ) , can not call with (  )  on it" ) ;+getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^817^^^^^815^818^+getClass (  ) .getName (  ) +" ) , can not call withArray (  )  on it" ) ;^[CLASS] JsonNode  [METHOD] withArray [RETURN_TYPE] JsonNode   String propertyName [VARIABLES] boolean  String  propertyName  
