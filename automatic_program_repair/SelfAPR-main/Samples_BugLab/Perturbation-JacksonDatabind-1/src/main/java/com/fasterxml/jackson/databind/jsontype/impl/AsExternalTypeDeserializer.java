[BugLab_Argument_Swapping]^super ( typePropertyName, idRes, bt, typeIdVisible, defaultImpl ) ;^25^^^^^22^26^super ( bt, idRes, typePropertyName, typeIdVisible, defaultImpl ) ;^[CLASS] AsExternalTypeDeserializer  [METHOD] <init> [RETURN_TYPE] Class)   JavaType bt TypeIdResolver idRes String typePropertyName boolean typeIdVisible Class<?> defaultImpl [VARIABLES] JavaType  bt  Class  defaultImpl  String  typePropertyName  boolean  typeIdVisible  long  serialVersionUID  TypeIdResolver  idRes  
[BugLab_Argument_Swapping]^super ( idRes, bt, typePropertyName, typeIdVisible, defaultImpl ) ;^25^^^^^22^26^super ( bt, idRes, typePropertyName, typeIdVisible, defaultImpl ) ;^[CLASS] AsExternalTypeDeserializer  [METHOD] <init> [RETURN_TYPE] Class)   JavaType bt TypeIdResolver idRes String typePropertyName boolean typeIdVisible Class<?> defaultImpl [VARIABLES] JavaType  bt  Class  defaultImpl  String  typePropertyName  boolean  typeIdVisible  long  serialVersionUID  TypeIdResolver  idRes  
[BugLab_Argument_Swapping]^super ( typeIdVisible, idRes, typePropertyName, bt, defaultImpl ) ;^25^^^^^22^26^super ( bt, idRes, typePropertyName, typeIdVisible, defaultImpl ) ;^[CLASS] AsExternalTypeDeserializer  [METHOD] <init> [RETURN_TYPE] Class)   JavaType bt TypeIdResolver idRes String typePropertyName boolean typeIdVisible Class<?> defaultImpl [VARIABLES] JavaType  bt  Class  defaultImpl  String  typePropertyName  boolean  typeIdVisible  long  serialVersionUID  TypeIdResolver  idRes  
[BugLab_Argument_Swapping]^super ( bt, idRes, typePropertyName, defaultImpl, typeIdVisible ) ;^25^^^^^22^26^super ( bt, idRes, typePropertyName, typeIdVisible, defaultImpl ) ;^[CLASS] AsExternalTypeDeserializer  [METHOD] <init> [RETURN_TYPE] Class)   JavaType bt TypeIdResolver idRes String typePropertyName boolean typeIdVisible Class<?> defaultImpl [VARIABLES] JavaType  bt  Class  defaultImpl  String  typePropertyName  boolean  typeIdVisible  long  serialVersionUID  TypeIdResolver  idRes  
[BugLab_Argument_Swapping]^super ( property, src ) ;^29^^^^^28^30^super ( src, property ) ;^[CLASS] AsExternalTypeDeserializer  [METHOD] <init> [RETURN_TYPE] BeanProperty)   AsExternalTypeDeserializer src BeanProperty property [VARIABLES] boolean  AsExternalTypeDeserializer  src  long  serialVersionUID  BeanProperty  property  
[BugLab_Argument_Swapping]^if  ( _property == prop )  {^35^^^^^33^39^if  ( prop == _property )  {^[CLASS] AsExternalTypeDeserializer  [METHOD] forProperty [RETURN_TYPE] TypeDeserializer   BeanProperty prop [VARIABLES] long  serialVersionUID  BeanProperty  prop  boolean  
[BugLab_Wrong_Operator]^if  ( prop != _property )  {^35^^^^^33^39^if  ( prop == _property )  {^[CLASS] AsExternalTypeDeserializer  [METHOD] forProperty [RETURN_TYPE] TypeDeserializer   BeanProperty prop [VARIABLES] long  serialVersionUID  BeanProperty  prop  boolean  
