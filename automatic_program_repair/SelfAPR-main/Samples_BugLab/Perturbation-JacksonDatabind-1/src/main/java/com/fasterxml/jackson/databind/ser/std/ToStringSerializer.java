[BugLab_Wrong_Operator]^if  ( value != null )  {^42^^^^^41^48^if  ( value == null )  {^[CLASS] ToStringSerializer  [METHOD] isEmpty [RETURN_TYPE] boolean   Object value [VARIABLES] ToStringSerializer  instance  Object  value  String  str  boolean  
[BugLab_Wrong_Literal]^return false;^43^^^^^41^48^return true;^[CLASS] ToStringSerializer  [METHOD] isEmpty [RETURN_TYPE] boolean   Object value [VARIABLES] ToStringSerializer  instance  Object  value  String  str  boolean  
[BugLab_Wrong_Operator]^return  ( str == null )  &&  ( str.length (  )  == 0 ) ;^47^^^^^41^48^return  ( str == null )  ||  ( str.length (  )  == 0 ) ;^[CLASS] ToStringSerializer  [METHOD] isEmpty [RETURN_TYPE] boolean   Object value [VARIABLES] ToStringSerializer  instance  Object  value  String  str  boolean  
[BugLab_Wrong_Operator]^return  ( str != null )  ||  ( str.length (  )  == 0 ) ;^47^^^^^41^48^return  ( str == null )  ||  ( str.length (  )  == 0 ) ;^[CLASS] ToStringSerializer  [METHOD] isEmpty [RETURN_TYPE] boolean   Object value [VARIABLES] ToStringSerializer  instance  Object  value  String  str  boolean  
[BugLab_Wrong_Operator]^return  ( str == null )  ||  ( str.length (  )  != 0 ) ;^47^^^^^41^48^return  ( str == null )  ||  ( str.length (  )  == 0 ) ;^[CLASS] ToStringSerializer  [METHOD] isEmpty [RETURN_TYPE] boolean   Object value [VARIABLES] ToStringSerializer  instance  Object  value  String  str  boolean  
[BugLab_Argument_Swapping]^typeSer.writeTypePrefixForScalar ( jgen, value ) ;^73^^^^^69^76^typeSer.writeTypePrefixForScalar ( value, jgen ) ;^[CLASS] ToStringSerializer  [METHOD] serializeWithType [RETURN_TYPE] void   Object value JsonGenerator jgen SerializerProvider provider TypeSerializer typeSer [VARIABLES] TypeSerializer  typeSer  JsonGenerator  jgen  ToStringSerializer  instance  Object  value  boolean  SerializerProvider  provider  
[BugLab_Argument_Swapping]^serialize ( jgen, value, provider ) ;^74^^^^^69^76^serialize ( value, jgen, provider ) ;^[CLASS] ToStringSerializer  [METHOD] serializeWithType [RETURN_TYPE] void   Object value JsonGenerator jgen SerializerProvider provider TypeSerializer typeSer [VARIABLES] TypeSerializer  typeSer  JsonGenerator  jgen  ToStringSerializer  instance  Object  value  boolean  SerializerProvider  provider  
[BugLab_Argument_Swapping]^serialize ( value, provider, jgen ) ;^74^^^^^69^76^serialize ( value, jgen, provider ) ;^[CLASS] ToStringSerializer  [METHOD] serializeWithType [RETURN_TYPE] void   Object value JsonGenerator jgen SerializerProvider provider TypeSerializer typeSer [VARIABLES] TypeSerializer  typeSer  JsonGenerator  jgen  ToStringSerializer  instance  Object  value  boolean  SerializerProvider  provider  
[BugLab_Argument_Swapping]^typeSer.writeTypeSuffixForScalar ( jgen, value ) ;^75^^^^^69^76^typeSer.writeTypeSuffixForScalar ( value, jgen ) ;^[CLASS] ToStringSerializer  [METHOD] serializeWithType [RETURN_TYPE] void   Object value JsonGenerator jgen SerializerProvider provider TypeSerializer typeSer [VARIABLES] TypeSerializer  typeSer  JsonGenerator  jgen  ToStringSerializer  instance  Object  value  boolean  SerializerProvider  provider  
[BugLab_Wrong_Literal]^return createSchemaNode ( "string", false ) ;^82^^^^^79^83^return createSchemaNode ( "string", true ) ;^[CLASS] ToStringSerializer  [METHOD] getSchema [RETURN_TYPE] JsonNode   SerializerProvider provider Type typeHint [VARIABLES] ToStringSerializer  instance  Type  typeHint  boolean  SerializerProvider  provider  
[BugLab_Wrong_Operator]^if  ( visitor == null )  {^89^^^^^86^92^if  ( visitor != null )  {^[CLASS] ToStringSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] ToStringSerializer  instance  JavaType  typeHint  boolean  JsonFormatVisitorWrapper  visitor  
