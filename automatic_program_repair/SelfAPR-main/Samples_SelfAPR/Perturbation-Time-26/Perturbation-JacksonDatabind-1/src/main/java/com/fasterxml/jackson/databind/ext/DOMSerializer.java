[P8_Replace_Mix]^protected  DOMImplementationLS _domImpl;^21^^^^^16^26^protected final DOMImplementationLS _domImpl;^[CLASS] DOMSerializer   [VARIABLES] 
[P14_Delete_Statement]^^25^^^^^23^33^super ( Node.class ) ;^[CLASS] DOMSerializer  [METHOD] <init> [RETURN_TYPE] DOMSerializer()   [VARIABLES] DOMImplementationRegistry  registry  boolean  Exception  e  DOMImplementationLS  _domImpl  
[P8_Replace_Mix]^registry  =  registry ;^28^^^^^23^33^registry = DOMImplementationRegistry.newInstance (  ) ;^[CLASS] DOMSerializer  [METHOD] <init> [RETURN_TYPE] DOMSerializer()   [VARIABLES] DOMImplementationRegistry  registry  boolean  Exception  e  DOMImplementationLS  _domImpl  
[P11_Insert_Donor_Statement]^_domImpl =  ( DOMImplementationLS ) registry.getDOMImplementation ( "LS" ) ;registry = DOMImplementationRegistry.newInstance (  ) ;^28^^^^^23^33^registry = DOMImplementationRegistry.newInstance (  ) ;^[CLASS] DOMSerializer  [METHOD] <init> [RETURN_TYPE] DOMSerializer()   [VARIABLES] DOMImplementationRegistry  registry  boolean  Exception  e  DOMImplementationLS  _domImpl  
[P14_Delete_Statement]^^28^^^^^23^33^registry = DOMImplementationRegistry.newInstance (  ) ;^[CLASS] DOMSerializer  [METHOD] <init> [RETURN_TYPE] DOMSerializer()   [VARIABLES] DOMImplementationRegistry  registry  boolean  Exception  e  DOMImplementationLS  _domImpl  
[P8_Replace_Mix]^return ;^30^^^^^23^33^throw new IllegalStateException  (" ")  ;^[CLASS] DOMSerializer  [METHOD] <init> [RETURN_TYPE] DOMSerializer()   [VARIABLES] DOMImplementationRegistry  registry  boolean  Exception  e  DOMImplementationLS  _domImpl  
[P14_Delete_Statement]^^30^^^^^23^33^throw new IllegalStateException  (" ")  ;^[CLASS] DOMSerializer  [METHOD] <init> [RETURN_TYPE] DOMSerializer()   [VARIABLES] DOMImplementationRegistry  registry  boolean  Exception  e  DOMImplementationLS  _domImpl  
[P3_Replace_Literal]^_domImpl =  ( DOMImplementationLS ) registry.getDOMImplementation ( "S" ) ;^32^^^^^23^33^_domImpl =  ( DOMImplementationLS ) registry.getDOMImplementation ( "LS" ) ;^[CLASS] DOMSerializer  [METHOD] <init> [RETURN_TYPE] DOMSerializer()   [VARIABLES] DOMImplementationRegistry  registry  boolean  Exception  e  DOMImplementationLS  _domImpl  
[P8_Replace_Mix]^_domImpl =   ( DOMImplementationLS ) null.getDOMImplementation ( "LS" ) ;^32^^^^^23^33^_domImpl =  ( DOMImplementationLS ) registry.getDOMImplementation ( "LS" ) ;^[CLASS] DOMSerializer  [METHOD] <init> [RETURN_TYPE] DOMSerializer()   [VARIABLES] DOMImplementationRegistry  registry  boolean  Exception  e  DOMImplementationLS  _domImpl  
[P11_Insert_Donor_Statement]^registry = DOMImplementationRegistry.newInstance (  ) ;_domImpl =  ( DOMImplementationLS ) registry.getDOMImplementation ( "LS" ) ;^32^^^^^23^33^_domImpl =  ( DOMImplementationLS ) registry.getDOMImplementation ( "LS" ) ;^[CLASS] DOMSerializer  [METHOD] <init> [RETURN_TYPE] DOMSerializer()   [VARIABLES] DOMImplementationRegistry  registry  boolean  Exception  e  DOMImplementationLS  _domImpl  
[P3_Replace_Literal]^_domImpl =  ( DOMImplementationLS ) registry.getDOMImplementation ( "L" ) ;^32^^^^^23^33^_domImpl =  ( DOMImplementationLS ) registry.getDOMImplementation ( "LS" ) ;^[CLASS] DOMSerializer  [METHOD] <init> [RETURN_TYPE] DOMSerializer()   [VARIABLES] DOMImplementationRegistry  registry  boolean  Exception  e  DOMImplementationLS  _domImpl  
[P8_Replace_Mix]^_domImpl =  ( DOMImplementationLS ) registry .newInstance (  )  ;^32^^^^^23^33^_domImpl =  ( DOMImplementationLS ) registry.getDOMImplementation ( "LS" ) ;^[CLASS] DOMSerializer  [METHOD] <init> [RETURN_TYPE] DOMSerializer()   [VARIABLES] DOMImplementationRegistry  registry  boolean  Exception  e  DOMImplementationLS  _domImpl  
[P14_Delete_Statement]^^32^^^^^23^33^_domImpl =  ( DOMImplementationLS ) registry.getDOMImplementation ( "LS" ) ;^[CLASS] DOMSerializer  [METHOD] <init> [RETURN_TYPE] DOMSerializer()   [VARIABLES] DOMImplementationRegistry  registry  boolean  Exception  e  DOMImplementationLS  _domImpl  
[P15_Unwrap_Block]^throw new java.lang.IllegalStateException("Could not find DOM LS");^39^40^41^42^^36^42^if   (" ")  ; LSSerializer writer = _domImpl.createLSSerializer (  ) ; jgen.writeString ( writer.writeToString ( value )  ) ; }^[CLASS] DOMSerializer  [METHOD] serialize [RETURN_TYPE] void   Node value JsonGenerator jgen SerializerProvider provider [VARIABLES] JsonGenerator  jgen  LSSerializer  writer  boolean  SerializerProvider  provider  DOMImplementationLS  _domImpl  Node  value  
[P16_Remove_Block]^^39^40^41^42^^36^42^if   (" ")  ; LSSerializer writer = _domImpl.createLSSerializer (  ) ; jgen.writeString ( writer.writeToString ( value )  ) ; }^[CLASS] DOMSerializer  [METHOD] serialize [RETURN_TYPE] void   Node value JsonGenerator jgen SerializerProvider provider [VARIABLES] JsonGenerator  jgen  LSSerializer  writer  boolean  SerializerProvider  provider  DOMImplementationLS  _domImpl  Node  value  
[P8_Replace_Mix]^return 0;^39^^^^^36^42^if   (" ")  ;^[CLASS] DOMSerializer  [METHOD] serialize [RETURN_TYPE] void   Node value JsonGenerator jgen SerializerProvider provider [VARIABLES] JsonGenerator  jgen  LSSerializer  writer  boolean  SerializerProvider  provider  DOMImplementationLS  _domImpl  Node  value  
[P14_Delete_Statement]^^40^^^^^36^42^LSSerializer writer = _domImpl.createLSSerializer (  ) ;^[CLASS] DOMSerializer  [METHOD] serialize [RETURN_TYPE] void   Node value JsonGenerator jgen SerializerProvider provider [VARIABLES] JsonGenerator  jgen  LSSerializer  writer  boolean  SerializerProvider  provider  DOMImplementationLS  _domImpl  Node  value  
[P5_Replace_Variable]^jgen.writeString ( value.writeToString ( writer )  ) ;^41^^^^^36^42^jgen.writeString ( writer.writeToString ( value )  ) ;^[CLASS] DOMSerializer  [METHOD] serialize [RETURN_TYPE] void   Node value JsonGenerator jgen SerializerProvider provider [VARIABLES] JsonGenerator  jgen  LSSerializer  writer  boolean  SerializerProvider  provider  DOMImplementationLS  _domImpl  Node  value  
[P14_Delete_Statement]^^41^^^^^36^42^jgen.writeString ( writer.writeToString ( value )  ) ;^[CLASS] DOMSerializer  [METHOD] serialize [RETURN_TYPE] void   Node value JsonGenerator jgen SerializerProvider provider [VARIABLES] JsonGenerator  jgen  LSSerializer  writer  boolean  SerializerProvider  provider  DOMImplementationLS  _domImpl  Node  value  
[P3_Replace_Literal]^return createSchemaNode ( "stringstrin", true ) ;^48^^^^^45^49^return createSchemaNode ( "string", true ) ;^[CLASS] DOMSerializer  [METHOD] getSchema [RETURN_TYPE] JsonNode   SerializerProvider provider Type typeHint [VARIABLES] Type  typeHint  boolean  SerializerProvider  provider  DOMImplementationLS  _domImpl  
[P3_Replace_Literal]^return createSchemaNode ( "string", false ) ;^48^^^^^45^49^return createSchemaNode ( "string", true ) ;^[CLASS] DOMSerializer  [METHOD] getSchema [RETURN_TYPE] JsonNode   SerializerProvider provider Type typeHint [VARIABLES] Type  typeHint  boolean  SerializerProvider  provider  DOMImplementationLS  _domImpl  
[P3_Replace_Literal]^return createSchemaNode ( "s", true ) ;^48^^^^^45^49^return createSchemaNode ( "string", true ) ;^[CLASS] DOMSerializer  [METHOD] getSchema [RETURN_TYPE] JsonNode   SerializerProvider provider Type typeHint [VARIABLES] Type  typeHint  boolean  SerializerProvider  provider  DOMImplementationLS  _domImpl  
[P14_Delete_Statement]^^48^^^^^45^49^return createSchemaNode ( "string", true ) ;^[CLASS] DOMSerializer  [METHOD] getSchema [RETURN_TYPE] JsonNode   SerializerProvider provider Type typeHint [VARIABLES] Type  typeHint  boolean  SerializerProvider  provider  DOMImplementationLS  _domImpl  
[P2_Replace_Operator]^if  ( visitor == null )  visitor.expectAnyFormat ( typeHint ) ;^55^^^^^52^56^if  ( visitor != null )  visitor.expectAnyFormat ( typeHint ) ;^[CLASS] DOMSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] JavaType  typeHint  boolean  JsonFormatVisitorWrapper  visitor  DOMImplementationLS  _domImpl  
[P5_Replace_Variable]^if  ( typeHint != null )  visitor.expectAnyFormat ( visitor ) ;^55^^^^^52^56^if  ( visitor != null )  visitor.expectAnyFormat ( typeHint ) ;^[CLASS] DOMSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] JavaType  typeHint  boolean  JsonFormatVisitorWrapper  visitor  DOMImplementationLS  _domImpl  
[P8_Replace_Mix]^if  ( visitor != true )  visitor.expectAnyFormat ( typeHint ) ;^55^^^^^52^56^if  ( visitor != null )  visitor.expectAnyFormat ( typeHint ) ;^[CLASS] DOMSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] JavaType  typeHint  boolean  JsonFormatVisitorWrapper  visitor  DOMImplementationLS  _domImpl  
[P15_Unwrap_Block]^visitor.expectAnyFormat(typeHint);^55^56^^^^52^56^if  ( visitor != null )  visitor.expectAnyFormat ( typeHint ) ; }^[CLASS] DOMSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] JavaType  typeHint  boolean  JsonFormatVisitorWrapper  visitor  DOMImplementationLS  _domImpl  
[P16_Remove_Block]^^55^56^^^^52^56^if  ( visitor != null )  visitor.expectAnyFormat ( typeHint ) ; }^[CLASS] DOMSerializer  [METHOD] acceptJsonFormatVisitor [RETURN_TYPE] void   JsonFormatVisitorWrapper visitor JavaType typeHint [VARIABLES] JavaType  typeHint  boolean  JsonFormatVisitorWrapper  visitor  DOMImplementationLS  _domImpl  
