[BugLab_Wrong_Operator]^if  ( src != null )  {^42^^^^^41^46^if  ( src == null )  {^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src [VARIABLES] MemoryRefStack  ancestors  Object  src  boolean  serializeNulls  ParameterizedTypeHandlerMap  serializers  ObjectNavigatorFactory  factory  
[BugLab_Wrong_Literal]^return serialize ( src, src.getClass (  ) , false ) ;^45^^^^^41^46^return serialize ( src, src.getClass (  ) , true ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src [VARIABLES] MemoryRefStack  ancestors  Object  src  boolean  serializeNulls  ParameterizedTypeHandlerMap  serializers  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^return serialize ( typeOfSrc, src, true ) ;^49^^^^^48^50^return serialize ( src, typeOfSrc, true ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc [VARIABLES] MemoryRefStack  ancestors  Object  src  Type  typeOfSrc  boolean  serializeNulls  ParameterizedTypeHandlerMap  serializers  ObjectNavigatorFactory  factory  
[BugLab_Wrong_Literal]^return serialize ( src, typeOfSrc, false ) ;^49^^^^^48^50^return serialize ( src, typeOfSrc, true ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc [VARIABLES] MemoryRefStack  ancestors  Object  src  Type  typeOfSrc  boolean  serializeNulls  ParameterizedTypeHandlerMap  serializers  ObjectNavigatorFactory  factory  
[BugLab_Variable_Misuse]^ObjectNavigator on = factory.create ( new ObjectTypePair ( src, typeOfSrc, serializeNulls )  ) ;^53^^^^^52^58^ObjectNavigator on = factory.create ( new ObjectTypePair ( src, typeOfSrc, preserveType )  ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^ObjectNavigator on = factory.create ( new ObjectTypePair ( preserveType, typeOfSrc, src )  ) ;^53^^^^^52^58^ObjectNavigator on = factory.create ( new ObjectTypePair ( src, typeOfSrc, preserveType )  ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^ObjectNavigator on = factory.create ( new ObjectTypePair ( typeOfSrc, src, preserveType )  ) ;^53^^^^^52^58^ObjectNavigator on = factory.create ( new ObjectTypePair ( src, typeOfSrc, preserveType )  ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^ObjectNavigator on = factory.create ( new ObjectTypePair ( src, preserveType, typeOfSrc )  ) ;^53^^^^^52^58^ObjectNavigator on = factory.create ( new ObjectTypePair ( src, typeOfSrc, preserveType )  ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^ObjectNavigator on = src.create ( new ObjectTypePair ( factory, typeOfSrc, preserveType )  ) ;^53^^^^^52^58^ObjectNavigator on = factory.create ( new ObjectTypePair ( src, typeOfSrc, preserveType )  ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^ObjectNavigator on = preserveType.create ( new ObjectTypePair ( src, typeOfSrc, factory )  ) ;^53^^^^^52^58^ObjectNavigator on = factory.create ( new ObjectTypePair ( src, typeOfSrc, preserveType )  ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^ObjectNavigator on = typeOfSrc.create ( new ObjectTypePair ( src, factory, preserveType )  ) ;^53^^^^^52^58^ObjectNavigator on = factory.create ( new ObjectTypePair ( src, typeOfSrc, preserveType )  ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Variable_Misuse]^new JsonSerializationVisitor ( factory, preserveType, serializers, this, ancestors ) ;^55^^^^^52^58^new JsonSerializationVisitor ( factory, serializeNulls, serializers, this, ancestors ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^new JsonSerializationVisitor ( ancestors, serializeNulls, serializers, this, factory ) ;^55^^^^^52^58^new JsonSerializationVisitor ( factory, serializeNulls, serializers, this, ancestors ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^new JsonSerializationVisitor ( factory, ancestors, serializers, this, serializeNulls ) ;^55^^^^^52^58^new JsonSerializationVisitor ( factory, serializeNulls, serializers, this, ancestors ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^new JsonSerializationVisitor ( factory, serializeNulls, ancestors, this, serializers ) ;^55^^^^^52^58^new JsonSerializationVisitor ( factory, serializeNulls, serializers, this, ancestors ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Variable_Misuse]^JsonSerializationVisitor visitor = new JsonSerializationVisitor ( factory, preserveType, serializers, this, ancestors ) ;^54^55^^^^52^58^JsonSerializationVisitor visitor = new JsonSerializationVisitor ( factory, serializeNulls, serializers, this, ancestors ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^JsonSerializationVisitor visitor = new JsonSerializationVisitor ( serializeNulls, factory, serializers, this, ancestors ) ;^54^55^^^^52^58^JsonSerializationVisitor visitor = new JsonSerializationVisitor ( factory, serializeNulls, serializers, this, ancestors ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^JsonSerializationVisitor visitor = new JsonSerializationVisitor ( factory, serializers, serializeNulls, this, ancestors ) ;^54^55^^^^52^58^JsonSerializationVisitor visitor = new JsonSerializationVisitor ( factory, serializeNulls, serializers, this, ancestors ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^JsonSerializationVisitor visitor = new JsonSerializationVisitor ( factory, serializeNulls, ancestors, this, serializers ) ;^54^55^^^^52^58^JsonSerializationVisitor visitor = new JsonSerializationVisitor ( factory, serializeNulls, serializers, this, ancestors ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
[BugLab_Argument_Swapping]^JsonSerializationVisitor visitor = new JsonSerializationVisitor ( factory, ancestors, serializers, this, serializeNulls ) ;^54^55^^^^52^58^JsonSerializationVisitor visitor = new JsonSerializationVisitor ( factory, serializeNulls, serializers, this, ancestors ) ;^[CLASS] JsonSerializationContextDefault  [METHOD] serialize [RETURN_TYPE] JsonElement   Object src Type typeOfSrc boolean preserveType [VARIABLES] Type  typeOfSrc  boolean  preserveType  serializeNulls  ObjectNavigator  on  MemoryRefStack  ancestors  Object  src  ParameterizedTypeHandlerMap  serializers  JsonSerializationVisitor  visitor  ObjectNavigatorFactory  factory  
