[BugLab_Wrong_Literal]^super ( Enum.class, true ) ;^68^^^^^66^71^super ( Enum.class, false ) ;^[CLASS] EnumSerializer  [METHOD] <init> [RETURN_TYPE] Boolean)   EnumValues v Boolean serializeAsIndex [VARIABLES] Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  boolean  
[BugLab_Variable_Misuse]^_values = _values;^69^^^^^66^71^_values = v;^[CLASS] EnumSerializer  [METHOD] <init> [RETURN_TYPE] Boolean)   EnumValues v Boolean serializeAsIndex [VARIABLES] Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  boolean  
[BugLab_Variable_Misuse]^_serializeAsIndex = _serializeAsIndex;^70^^^^^66^71^_serializeAsIndex = serializeAsIndex;^[CLASS] EnumSerializer  [METHOD] <init> [RETURN_TYPE] Boolean)   EnumValues v Boolean serializeAsIndex [VARIABLES] Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  boolean  
[BugLab_Argument_Swapping]^EnumValues v = intr.isEnabled ( SerializationFeature.WRITE_ENUMS_USING_TO_STRING ) ? EnumValues.constructFromToString ( enumClass, config )  : EnumValues.constructFromName ( enumClass, intr ) ;^84^85^^^^79^88^EnumValues v = config.isEnabled ( SerializationFeature.WRITE_ENUMS_USING_TO_STRING ) ? EnumValues.constructFromToString ( enumClass, intr )  : EnumValues.constructFromName ( enumClass, intr ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc Value format [VARIABLES] boolean  SerializationConfig  config  BeanDescription  beanDesc  Value  format  AnnotationIntrospector  intr  Class  enumClass  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Argument_Swapping]^EnumValues v = config.isEnabled ( SerializationFeature.WRITE_ENUMS_USING_TO_STRING ) ? EnumValues.constructFromToString ( intr, enumClass )  : EnumValues.constructFromName ( enumClass, intr ) ;^84^85^^^^79^88^EnumValues v = config.isEnabled ( SerializationFeature.WRITE_ENUMS_USING_TO_STRING ) ? EnumValues.constructFromToString ( enumClass, intr )  : EnumValues.constructFromName ( enumClass, intr ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc Value format [VARIABLES] boolean  SerializationConfig  config  BeanDescription  beanDesc  Value  format  AnnotationIntrospector  intr  Class  enumClass  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Argument_Swapping]^? EnumValues.constructFromToString ( intr, enumClass )  : EnumValues.constructFromName ( enumClass, intr ) ;^85^^^^^79^88^? EnumValues.constructFromToString ( enumClass, intr )  : EnumValues.constructFromName ( enumClass, intr ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc Value format [VARIABLES] boolean  SerializationConfig  config  BeanDescription  beanDesc  Value  format  AnnotationIntrospector  intr  Class  enumClass  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Variable_Misuse]^Boolean serializeAsIndex = _isShapeWrittenUsingIndex ( null, format, true ) ;^86^^^^^79^88^Boolean serializeAsIndex = _isShapeWrittenUsingIndex ( enumClass, format, true ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc Value format [VARIABLES] boolean  SerializationConfig  config  BeanDescription  beanDesc  Value  format  AnnotationIntrospector  intr  Class  enumClass  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Argument_Swapping]^Boolean serializeAsIndex = _isShapeWrittenUsingIndex ( format, enumClass, true ) ;^86^^^^^79^88^Boolean serializeAsIndex = _isShapeWrittenUsingIndex ( enumClass, format, true ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc Value format [VARIABLES] boolean  SerializationConfig  config  BeanDescription  beanDesc  Value  format  AnnotationIntrospector  intr  Class  enumClass  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Literal]^Boolean serializeAsIndex = _isShapeWrittenUsingIndex ( enumClass, format, false ) ;^86^^^^^79^88^Boolean serializeAsIndex = _isShapeWrittenUsingIndex ( enumClass, format, true ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc Value format [VARIABLES] boolean  SerializationConfig  config  BeanDescription  beanDesc  Value  format  AnnotationIntrospector  intr  Class  enumClass  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Variable_Misuse]^Boolean serializeAsIndex = _isShapeWrittenUsingIndex ( enumClass, null, true ) ;^86^^^^^79^88^Boolean serializeAsIndex = _isShapeWrittenUsingIndex ( enumClass, format, true ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc Value format [VARIABLES] boolean  SerializationConfig  config  BeanDescription  beanDesc  Value  format  AnnotationIntrospector  intr  Class  enumClass  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Variable_Misuse]^return new EnumSerializer ( _values, serializeAsIndex ) ;^87^^^^^79^88^return new EnumSerializer ( v, serializeAsIndex ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc Value format [VARIABLES] boolean  SerializationConfig  config  BeanDescription  beanDesc  Value  format  AnnotationIntrospector  intr  Class  enumClass  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Variable_Misuse]^return new EnumSerializer ( v, _serializeAsIndex ) ;^87^^^^^79^88^return new EnumSerializer ( v, serializeAsIndex ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc Value format [VARIABLES] boolean  SerializationConfig  config  BeanDescription  beanDesc  Value  format  AnnotationIntrospector  intr  Class  enumClass  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Argument_Swapping]^return new EnumSerializer ( serializeAsIndex, v ) ;^87^^^^^79^88^return new EnumSerializer ( v, serializeAsIndex ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc Value format [VARIABLES] boolean  SerializationConfig  config  BeanDescription  beanDesc  Value  format  AnnotationIntrospector  intr  Class  enumClass  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Variable_Misuse]^return construct ( null, config, beanDesc, beanDesc.findExpectedFormat ( null )  ) ;^97^^^^^94^98^return construct ( enumClass, config, beanDesc, beanDesc.findExpectedFormat ( null )  ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc [VARIABLES] Class  enumClass  boolean  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  SerializationConfig  config  BeanDescription  beanDesc  
[BugLab_Argument_Swapping]^return construct ( config, enumClass, beanDesc, beanDesc.findExpectedFormat ( null )  ) ;^97^^^^^94^98^return construct ( enumClass, config, beanDesc, beanDesc.findExpectedFormat ( null )  ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc [VARIABLES] Class  enumClass  boolean  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  SerializationConfig  config  BeanDescription  beanDesc  
[BugLab_Argument_Swapping]^return construct ( enumClass, beanDesc, config, beanDesc.findExpectedFormat ( null )  ) ;^97^^^^^94^98^return construct ( enumClass, config, beanDesc, beanDesc.findExpectedFormat ( null )  ) ;^[CLASS] EnumSerializer  [METHOD] construct [RETURN_TYPE] EnumSerializer   Enum<?>> enumClass SerializationConfig config BeanDescription beanDesc [VARIABLES] Class  enumClass  boolean  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  SerializationConfig  config  BeanDescription  beanDesc  
[BugLab_Wrong_Operator]^if  ( property == null )  {^109^^^^^106^119^if  ( property != null )  {^[CLASS] EnumSerializer  [METHOD] createContextual [RETURN_TYPE] JsonSerializer   SerializerProvider prov BeanProperty property [VARIABLES] Value  format  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  prov  EnumValues  _values  v  BeanProperty  property  
[BugLab_Wrong_Operator]^if  ( format == null )  {^111^^^^^106^119^if  ( format != null )  {^[CLASS] EnumSerializer  [METHOD] createContextual [RETURN_TYPE] JsonSerializer   SerializerProvider prov BeanProperty property [VARIABLES] Value  format  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  prov  EnumValues  _values  v  BeanProperty  property  
[BugLab_Argument_Swapping]^if  ( _serializeAsIndex != serializeAsIndex )  {^113^^^^^106^119^if  ( serializeAsIndex != _serializeAsIndex )  {^[CLASS] EnumSerializer  [METHOD] createContextual [RETURN_TYPE] JsonSerializer   SerializerProvider prov BeanProperty property [VARIABLES] Value  format  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  prov  EnumValues  _values  v  BeanProperty  property  
[BugLab_Wrong_Operator]^if  ( serializeAsIndex == _serializeAsIndex )  {^113^^^^^106^119^if  ( serializeAsIndex != _serializeAsIndex )  {^[CLASS] EnumSerializer  [METHOD] createContextual [RETURN_TYPE] JsonSerializer   SerializerProvider prov BeanProperty property [VARIABLES] Value  format  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  prov  EnumValues  _values  v  BeanProperty  property  
[BugLab_Variable_Misuse]^return new EnumSerializer ( _values, _serializeAsIndex ) ;^114^^^^^106^119^return new EnumSerializer ( _values, serializeAsIndex ) ;^[CLASS] EnumSerializer  [METHOD] createContextual [RETURN_TYPE] JsonSerializer   SerializerProvider prov BeanProperty property [VARIABLES] Value  format  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  prov  EnumValues  _values  v  BeanProperty  property  
[BugLab_Variable_Misuse]^return new EnumSerializer ( v, serializeAsIndex ) ;^114^^^^^106^119^return new EnumSerializer ( _values, serializeAsIndex ) ;^[CLASS] EnumSerializer  [METHOD] createContextual [RETURN_TYPE] JsonSerializer   SerializerProvider prov BeanProperty property [VARIABLES] Value  format  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  prov  EnumValues  _values  v  BeanProperty  property  
[BugLab_Argument_Swapping]^return new EnumSerializer ( serializeAsIndex, _values ) ;^114^^^^^106^119^return new EnumSerializer ( _values, serializeAsIndex ) ;^[CLASS] EnumSerializer  [METHOD] createContextual [RETURN_TYPE] JsonSerializer   SerializerProvider prov BeanProperty property [VARIABLES] Value  format  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  prov  EnumValues  _values  v  BeanProperty  property  
[BugLab_Argument_Swapping]^Boolean serializeAsIndex = _isShapeWrittenUsingIndex ( format.getType (  ) .getRawClass (  ) , property, false ) ;^112^^^^^106^119^Boolean serializeAsIndex = _isShapeWrittenUsingIndex ( property.getType (  ) .getRawClass (  ) , format, false ) ;^[CLASS] EnumSerializer  [METHOD] createContextual [RETURN_TYPE] JsonSerializer   SerializerProvider prov BeanProperty property [VARIABLES] Value  format  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  prov  EnumValues  _values  v  BeanProperty  property  
[BugLab_Wrong_Literal]^Boolean serializeAsIndex = _isShapeWrittenUsingIndex ( property.getType (  ) .getRawClass (  ) , format, true ) ;^112^^^^^106^119^Boolean serializeAsIndex = _isShapeWrittenUsingIndex ( property.getType (  ) .getRawClass (  ) , format, false ) ;^[CLASS] EnumSerializer  [METHOD] createContextual [RETURN_TYPE] JsonSerializer   SerializerProvider prov BeanProperty property [VARIABLES] Value  format  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  prov  EnumValues  _values  v  BeanProperty  property  
[BugLab_Argument_Swapping]^JsonFormat.Value format = property.getAnnotationIntrospector (  ) .findFormat (  ( Annotated )  prov.getMember (  )  ) ;^110^^^^^106^119^JsonFormat.Value format = prov.getAnnotationIntrospector (  ) .findFormat (  ( Annotated )  property.getMember (  )  ) ;^[CLASS] EnumSerializer  [METHOD] createContextual [RETURN_TYPE] JsonSerializer   SerializerProvider prov BeanProperty property [VARIABLES] Value  format  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  prov  EnumValues  _values  v  BeanProperty  property  
[BugLab_Variable_Misuse]^if  ( _serializeAsIndex != _serializeAsIndex )  {^113^^^^^106^119^if  ( serializeAsIndex != _serializeAsIndex )  {^[CLASS] EnumSerializer  [METHOD] createContextual [RETURN_TYPE] JsonSerializer   SerializerProvider prov BeanProperty property [VARIABLES] Value  format  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  prov  EnumValues  _values  v  BeanProperty  property  
[BugLab_Variable_Misuse]^if  ( serializeAsIndex != serializeAsIndex )  {^113^^^^^106^119^if  ( serializeAsIndex != _serializeAsIndex )  {^[CLASS] EnumSerializer  [METHOD] createContextual [RETURN_TYPE] JsonSerializer   SerializerProvider prov BeanProperty property [VARIABLES] Value  format  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  prov  EnumValues  _values  v  BeanProperty  property  
[BugLab_Variable_Misuse]^public EnumValues getEnumValues (  )  { return v; }^127^^^^^122^132^public EnumValues getEnumValues (  )  { return _values; }^[CLASS] EnumSerializer  [METHOD] getEnumValues [RETURN_TYPE] EnumValues   [VARIABLES] Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  boolean  
[BugLab_Variable_Misuse]^jgen.writeString ( _values.serializedValueFor ( 1 )  ) ;^144^^^^^136^145^jgen.writeString ( _values.serializedValueFor ( en )  ) ;^[CLASS] EnumSerializer  [METHOD] serialize [RETURN_TYPE] void   Enum<?> en JsonGenerator jgen SerializerProvider provider [VARIABLES] Enum  en  JsonGenerator  jgen  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  provider  EnumValues  _values  v  
[BugLab_Variable_Misuse]^jgen.writeString ( v.serializedValueFor ( en )  ) ;^144^^^^^136^145^jgen.writeString ( _values.serializedValueFor ( en )  ) ;^[CLASS] EnumSerializer  [METHOD] serialize [RETURN_TYPE] void   Enum<?> en JsonGenerator jgen SerializerProvider provider [VARIABLES] Enum  en  JsonGenerator  jgen  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  provider  EnumValues  _values  v  
[BugLab_Variable_Misuse]^jgen.writeString ( _values.serializedValueFor ( this )  ) ;^144^^^^^136^145^jgen.writeString ( _values.serializedValueFor ( en )  ) ;^[CLASS] EnumSerializer  [METHOD] serialize [RETURN_TYPE] void   Enum<?> en JsonGenerator jgen SerializerProvider provider [VARIABLES] Enum  en  JsonGenerator  jgen  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  provider  EnumValues  _values  v  
[BugLab_Argument_Swapping]^jgen.writeString ( en.serializedValueFor ( _values )  ) ;^144^^^^^136^145^jgen.writeString ( _values.serializedValueFor ( en )  ) ;^[CLASS] EnumSerializer  [METHOD] serialize [RETURN_TYPE] void   Enum<?> en JsonGenerator jgen SerializerProvider provider [VARIABLES] Enum  en  JsonGenerator  jgen  boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  provider  EnumValues  _values  v  
[BugLab_Wrong_Literal]^return createSchemaNode ( "integer", false ) ;^152^^^^^148^165^return createSchemaNode ( "integer", true ) ;^[CLASS] EnumSerializer  [METHOD] getSchema [RETURN_TYPE] JsonNode   SerializerProvider provider Type typeHint [VARIABLES] SerializedString  value  Type  typeHint  JavaType  type  boolean  ObjectNode  objectNode  ArrayNode  enumNode  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  provider  EnumValues  _values  v  
[BugLab_Wrong_Literal]^ObjectNode objectNode = createSchemaNode ( "string", false ) ;^154^^^^^148^165^ObjectNode objectNode = createSchemaNode ( "string", true ) ;^[CLASS] EnumSerializer  [METHOD] getSchema [RETURN_TYPE] JsonNode   SerializerProvider provider Type typeHint [VARIABLES] SerializedString  value  Type  typeHint  JavaType  type  boolean  ObjectNode  objectNode  ArrayNode  enumNode  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  provider  EnumValues  _values  v  
[BugLab_Wrong_Operator]^if  ( typeHint == null )  {^155^^^^^148^165^if  ( typeHint != null )  {^[CLASS] EnumSerializer  [METHOD] getSchema [RETURN_TYPE] JsonNode   SerializerProvider provider Type typeHint [VARIABLES] SerializedString  value  Type  typeHint  JavaType  type  boolean  ObjectNode  objectNode  ArrayNode  enumNode  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  provider  EnumValues  _values  v  
[BugLab_Variable_Misuse]^for  ( SerializedString value : v.values (  )  )  {^159^^^^^148^165^for  ( SerializedString value : _values.values (  )  )  {^[CLASS] EnumSerializer  [METHOD] getSchema [RETURN_TYPE] JsonNode   SerializerProvider provider Type typeHint [VARIABLES] SerializedString  value  Type  typeHint  JavaType  type  boolean  ObjectNode  objectNode  ArrayNode  enumNode  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  provider  EnumValues  _values  v  
[BugLab_Argument_Swapping]^JavaType type = typeHint.constructType ( provider ) ;^156^^^^^148^165^JavaType type = provider.constructType ( typeHint ) ;^[CLASS] EnumSerializer  [METHOD] getSchema [RETURN_TYPE] JsonNode   SerializerProvider provider Type typeHint [VARIABLES] SerializedString  value  Type  typeHint  JavaType  type  boolean  ObjectNode  objectNode  ArrayNode  enumNode  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  provider  EnumValues  _values  v  
[BugLab_Wrong_Operator]^if  ( typeHint != null || stringVisitor != null )  {^179^^^^^168^189^if  ( typeHint != null && stringVisitor != null )  {^[CLASS] EnumSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] SerializedString  value  JavaType  typeHint  Set  enums  boolean  JsonFormatVisitorWrapper  visitor  JsonIntegerFormatVisitor  v2  Boolean  _serializeAsIndex  serializeAsIndex  JsonStringFormatVisitor  stringVisitor  EnumValues  _values  v  
[BugLab_Wrong_Operator]^if  ( typeHint == null && stringVisitor != null )  {^179^^^^^168^189^if  ( typeHint != null && stringVisitor != null )  {^[CLASS] EnumSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] SerializedString  value  JavaType  typeHint  Set  enums  boolean  JsonFormatVisitorWrapper  visitor  JsonIntegerFormatVisitor  v2  Boolean  _serializeAsIndex  serializeAsIndex  JsonStringFormatVisitor  stringVisitor  EnumValues  _values  v  
[BugLab_Wrong_Operator]^if  ( typeHint != null && stringVisitor == null )  {^179^^^^^168^189^if  ( typeHint != null && stringVisitor != null )  {^[CLASS] EnumSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] SerializedString  value  JavaType  typeHint  Set  enums  boolean  JsonFormatVisitorWrapper  visitor  JsonIntegerFormatVisitor  v2  Boolean  _serializeAsIndex  serializeAsIndex  JsonStringFormatVisitor  stringVisitor  EnumValues  _values  v  
[BugLab_Variable_Misuse]^stringVisitor.enumTypes ( null ) ;^185^^^^^168^189^stringVisitor.enumTypes ( enums ) ;^[CLASS] EnumSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] SerializedString  value  JavaType  typeHint  Set  enums  boolean  JsonFormatVisitorWrapper  visitor  JsonIntegerFormatVisitor  v2  Boolean  _serializeAsIndex  serializeAsIndex  JsonStringFormatVisitor  stringVisitor  EnumValues  _values  v  
[BugLab_Variable_Misuse]^for  ( SerializedString value : v.values (  )  )  {^182^^^^^168^189^for  ( SerializedString value : _values.values (  )  )  {^[CLASS] EnumSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] SerializedString  value  JavaType  typeHint  Set  enums  boolean  JsonFormatVisitorWrapper  visitor  JsonIntegerFormatVisitor  v2  Boolean  _serializeAsIndex  serializeAsIndex  JsonStringFormatVisitor  stringVisitor  EnumValues  _values  v  
[BugLab_Argument_Swapping]^JsonStringFormatVisitor stringVisitor = typeHint.expectStringFormat ( visitor ) ;^178^^^^^168^189^JsonStringFormatVisitor stringVisitor = visitor.expectStringFormat ( typeHint ) ;^[CLASS] EnumSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] SerializedString  value  JavaType  typeHint  Set  enums  boolean  JsonFormatVisitorWrapper  visitor  JsonIntegerFormatVisitor  v2  Boolean  _serializeAsIndex  serializeAsIndex  JsonStringFormatVisitor  stringVisitor  EnumValues  _values  v  
[BugLab_Wrong_Operator]^if  ( v2 == null )  {^174^^^^^168^189^if  ( v2 != null )  {^[CLASS] EnumSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] SerializedString  value  JavaType  typeHint  Set  enums  boolean  JsonFormatVisitorWrapper  visitor  JsonIntegerFormatVisitor  v2  Boolean  _serializeAsIndex  serializeAsIndex  JsonStringFormatVisitor  stringVisitor  EnumValues  _values  v  
[BugLab_Argument_Swapping]^JsonIntegerFormatVisitor v2 = typeHint.expectIntegerFormat ( visitor ) ;^173^^^^^168^189^JsonIntegerFormatVisitor v2 = visitor.expectIntegerFormat ( typeHint ) ;^[CLASS] EnumSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] SerializedString  value  JavaType  typeHint  Set  enums  boolean  JsonFormatVisitorWrapper  visitor  JsonIntegerFormatVisitor  v2  Boolean  _serializeAsIndex  serializeAsIndex  JsonStringFormatVisitor  stringVisitor  EnumValues  _values  v  
[BugLab_Variable_Misuse]^stringVisitor.enumTypes ( 4 ) ;^185^^^^^168^189^stringVisitor.enumTypes ( enums ) ;^[CLASS] EnumSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] SerializedString  value  JavaType  typeHint  Set  enums  boolean  JsonFormatVisitorWrapper  visitor  JsonIntegerFormatVisitor  v2  Boolean  _serializeAsIndex  serializeAsIndex  JsonStringFormatVisitor  stringVisitor  EnumValues  _values  v  
[BugLab_Variable_Misuse]^if  ( serializeAsIndex != null )  {^199^^^^^197^204^if  ( _serializeAsIndex != null )  {^[CLASS] EnumSerializer  [METHOD] _serializeAsIndex [RETURN_TYPE] boolean   SerializerProvider provider [VARIABLES] boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  provider  EnumValues  _values  v  
[BugLab_Wrong_Operator]^if  ( _serializeAsIndex == null )  {^199^^^^^197^204^if  ( _serializeAsIndex != null )  {^[CLASS] EnumSerializer  [METHOD] _serializeAsIndex [RETURN_TYPE] boolean   SerializerProvider provider [VARIABLES] boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  provider  EnumValues  _values  v  
[BugLab_Variable_Misuse]^return serializeAsIndex.booleanValue (  ) ;^200^^^^^197^204^return _serializeAsIndex.booleanValue (  ) ;^[CLASS] EnumSerializer  [METHOD] _serializeAsIndex [RETURN_TYPE] boolean   SerializerProvider provider [VARIABLES] boolean  Boolean  _serializeAsIndex  serializeAsIndex  SerializerProvider  provider  EnumValues  _values  v  
[BugLab_Wrong_Operator]^JsonFormat.Shape shape =  ( format != null )  ? null : format.getShape (  ) ;^212^^^^^209^229^JsonFormat.Shape shape =  ( format == null )  ? null : format.getShape (  ) ;^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^if  ( shape != null )  {^213^^^^^209^229^if  ( shape == null )  {^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^if  ( shape == Shape.ANY && shape == Shape.SCALAR )  {^216^^^^^209^229^if  ( shape == Shape.ANY || shape == Shape.SCALAR )  {^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^if  ( shape <= Shape.ANY || shape == Shape.SCALAR )  {^216^^^^^209^229^if  ( shape == Shape.ANY || shape == Shape.SCALAR )  {^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^if  ( shape != Shape.ANY || shape == Shape.SCALAR )  {^216^^^^^209^229^if  ( shape == Shape.ANY || shape == Shape.SCALAR )  {^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^if  ( shape > Shape.STRING )  {^219^^^^^209^229^if  ( shape == Shape.STRING )  {^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  &&  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^225^226^227^228^^209^229^throw new IllegalArgumentException  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  ==  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^225^226^227^228^^209^229^throw new IllegalArgumentException  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  ||  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^225^226^227^228^^209^229^throw new IllegalArgumentException  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  |  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^225^226^227^228^^209^229^throw new IllegalArgumentException  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  !=  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^225^226^227^228^^209^229^throw new IllegalArgumentException  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  >>  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^225^226^227^228^^209^229^throw new IllegalArgumentException  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException   instanceof   (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^225^226^227^228^^209^229^throw new IllegalArgumentException  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  <  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^225^226^227^228^^209^229^throw new IllegalArgumentException  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  >=  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^225^226^227^228^^209^229^throw new IllegalArgumentException  (" ") +", not supported as " +  ( fromClass? "class" : "property" ) +" annotation" ) ;^[CLASS] EnumSerializer  [METHOD] _isShapeWrittenUsingIndex [RETURN_TYPE] Boolean   Class<?> enumClass Value format boolean fromClass [VARIABLES] Value  format  Class  enumClass  boolean  fromClass  Shape  shape  Boolean  _serializeAsIndex  serializeAsIndex  EnumValues  _values  v  
