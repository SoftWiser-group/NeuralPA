[BugLab_Argument_Swapping]^super ( typeDeser, type, propDef, contextAnnotations ) ;^41^^^^^38^44^super ( propDef, type, typeDeser, contextAnnotations ) ;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] AnnotatedMethod)   BeanPropertyDefinition propDef JavaType type TypeDeserializer typeDeser Annotations contextAnnotations AnnotatedMethod method [VARIABLES] JavaType  type  Annotations  contextAnnotations  boolean  AnnotatedMethod  _annotated  method  BeanPropertyDefinition  propDef  Method  _getter  TypeDeserializer  typeDeser  long  serialVersionUID  
[BugLab_Argument_Swapping]^super ( type, propDef, typeDeser, contextAnnotations ) ;^41^^^^^38^44^super ( propDef, type, typeDeser, contextAnnotations ) ;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] AnnotatedMethod)   BeanPropertyDefinition propDef JavaType type TypeDeserializer typeDeser Annotations contextAnnotations AnnotatedMethod method [VARIABLES] JavaType  type  Annotations  contextAnnotations  boolean  AnnotatedMethod  _annotated  method  BeanPropertyDefinition  propDef  Method  _getter  TypeDeserializer  typeDeser  long  serialVersionUID  
[BugLab_Argument_Swapping]^super ( contextAnnotations, type, typeDeser, propDef ) ;^41^^^^^38^44^super ( propDef, type, typeDeser, contextAnnotations ) ;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] AnnotatedMethod)   BeanPropertyDefinition propDef JavaType type TypeDeserializer typeDeser Annotations contextAnnotations AnnotatedMethod method [VARIABLES] JavaType  type  Annotations  contextAnnotations  boolean  AnnotatedMethod  _annotated  method  BeanPropertyDefinition  propDef  Method  _getter  TypeDeserializer  typeDeser  long  serialVersionUID  
[BugLab_Variable_Misuse]^_annotated = _annotated;^42^^^^^38^44^_annotated = method;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] AnnotatedMethod)   BeanPropertyDefinition propDef JavaType type TypeDeserializer typeDeser Annotations contextAnnotations AnnotatedMethod method [VARIABLES] JavaType  type  Annotations  contextAnnotations  boolean  AnnotatedMethod  _annotated  method  BeanPropertyDefinition  propDef  Method  _getter  TypeDeserializer  typeDeser  long  serialVersionUID  
[BugLab_Variable_Misuse]^_getter = _annotated.getAnnotated (  ) ;^43^^^^^38^44^_getter = method.getAnnotated (  ) ;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] AnnotatedMethod)   BeanPropertyDefinition propDef JavaType type TypeDeserializer typeDeser Annotations contextAnnotations AnnotatedMethod method [VARIABLES] JavaType  type  Annotations  contextAnnotations  boolean  AnnotatedMethod  _annotated  method  BeanPropertyDefinition  propDef  Method  _getter  TypeDeserializer  typeDeser  long  serialVersionUID  
[BugLab_Argument_Swapping]^super ( deser, src ) ;^47^^^^^46^50^super ( src, deser ) ;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   SetterlessProperty src JsonDeserializer<?> deser [VARIABLES] SetterlessProperty  src  Method  _getter  boolean  AnnotatedMethod  _annotated  method  JsonDeserializer  deser  long  serialVersionUID  
[BugLab_Variable_Misuse]^_annotated = method;^48^^^^^46^50^_annotated = src._annotated;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   SetterlessProperty src JsonDeserializer<?> deser [VARIABLES] SetterlessProperty  src  Method  _getter  boolean  AnnotatedMethod  _annotated  method  JsonDeserializer  deser  long  serialVersionUID  
[BugLab_Argument_Swapping]^_annotated = src._annotated._annotated;^48^^^^^46^50^_annotated = src._annotated;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   SetterlessProperty src JsonDeserializer<?> deser [VARIABLES] SetterlessProperty  src  Method  _getter  boolean  AnnotatedMethod  _annotated  method  JsonDeserializer  deser  long  serialVersionUID  
[BugLab_Argument_Swapping]^_annotated = src;^48^^^^^46^50^_annotated = src._annotated;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   SetterlessProperty src JsonDeserializer<?> deser [VARIABLES] SetterlessProperty  src  Method  _getter  boolean  AnnotatedMethod  _annotated  method  JsonDeserializer  deser  long  serialVersionUID  
[BugLab_Variable_Misuse]^_getter = _getter;^49^^^^^46^50^_getter = src._getter;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   SetterlessProperty src JsonDeserializer<?> deser [VARIABLES] SetterlessProperty  src  Method  _getter  boolean  AnnotatedMethod  _annotated  method  JsonDeserializer  deser  long  serialVersionUID  
[BugLab_Argument_Swapping]^_getter = src._getter._getter;^49^^^^^46^50^_getter = src._getter;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   SetterlessProperty src JsonDeserializer<?> deser [VARIABLES] SetterlessProperty  src  Method  _getter  boolean  AnnotatedMethod  _annotated  method  JsonDeserializer  deser  long  serialVersionUID  
[BugLab_Argument_Swapping]^_getter = src;^49^^^^^46^50^_getter = src._getter;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   SetterlessProperty src JsonDeserializer<?> deser [VARIABLES] SetterlessProperty  src  Method  _getter  boolean  AnnotatedMethod  _annotated  method  JsonDeserializer  deser  long  serialVersionUID  
[BugLab_Argument_Swapping]^super ( newName, src ) ;^53^^^^^52^56^super ( src, newName ) ;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] String)   SetterlessProperty src String newName [VARIABLES] SetterlessProperty  src  Method  _getter  String  newName  boolean  AnnotatedMethod  _annotated  method  long  serialVersionUID  
[BugLab_Argument_Swapping]^_annotated = src._annotated._annotated;^54^^^^^52^56^_annotated = src._annotated;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] String)   SetterlessProperty src String newName [VARIABLES] SetterlessProperty  src  Method  _getter  String  newName  boolean  AnnotatedMethod  _annotated  method  long  serialVersionUID  
[BugLab_Argument_Swapping]^_annotated = src;^54^^^^^52^56^_annotated = src._annotated;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] String)   SetterlessProperty src String newName [VARIABLES] SetterlessProperty  src  Method  _getter  String  newName  boolean  AnnotatedMethod  _annotated  method  long  serialVersionUID  
[BugLab_Variable_Misuse]^_getter = _getter;^55^^^^^52^56^_getter = src._getter;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] String)   SetterlessProperty src String newName [VARIABLES] SetterlessProperty  src  Method  _getter  String  newName  boolean  AnnotatedMethod  _annotated  method  long  serialVersionUID  
[BugLab_Argument_Swapping]^_getter = src._getter._getter;^55^^^^^52^56^_getter = src._getter;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] String)   SetterlessProperty src String newName [VARIABLES] SetterlessProperty  src  Method  _getter  String  newName  boolean  AnnotatedMethod  _annotated  method  long  serialVersionUID  
[BugLab_Argument_Swapping]^_getter = src;^55^^^^^52^56^_getter = src._getter;^[CLASS] SetterlessProperty  [METHOD] <init> [RETURN_TYPE] String)   SetterlessProperty src String newName [VARIABLES] SetterlessProperty  src  Method  _getter  String  newName  boolean  AnnotatedMethod  _annotated  method  long  serialVersionUID  
[BugLab_Variable_Misuse]^return method.getAnnotation ( acls ) ;^76^^^^^75^77^return _annotated.getAnnotation ( acls ) ;^[CLASS] SetterlessProperty  [METHOD] getAnnotation [RETURN_TYPE] <A   Class<A> acls [VARIABLES] Class  acls  Method  _getter  boolean  AnnotatedMethod  _annotated  method  long  serialVersionUID  
[BugLab_Argument_Swapping]^return acls.getAnnotation ( _annotated ) ;^76^^^^^75^77^return _annotated.getAnnotation ( acls ) ;^[CLASS] SetterlessProperty  [METHOD] getAnnotation [RETURN_TYPE] <A   Class<A> acls [VARIABLES] Class  acls  Method  _getter  boolean  AnnotatedMethod  _annotated  method  long  serialVersionUID  
[BugLab_Variable_Misuse]^@Override public AnnotatedMember getMember (  )  {  return method; }^79^^^^^74^84^@Override public AnnotatedMember getMember (  )  {  return _annotated; }^[CLASS] SetterlessProperty  [METHOD] getMember [RETURN_TYPE] AnnotatedMember   [VARIABLES] Method  _getter  boolean  AnnotatedMethod  _annotated  method  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( t != JsonToken.VALUE_NULL )  {^93^^^^^91^117^if  ( t == JsonToken.VALUE_NULL )  {^[CLASS] SetterlessProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] boolean  AnnotatedMethod  _annotated  method  DeserializationContext  ctxt  Object  instance  toModify  JsonToken  t  Method  _getter  long  serialVersionUID  Exception  e  JsonParser  jp  
[BugLab_Variable_Misuse]^toModify = _getter.invoke ( toModify ) ;^103^^^^^91^117^toModify = _getter.invoke ( instance ) ;^[CLASS] SetterlessProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] boolean  AnnotatedMethod  _annotated  method  DeserializationContext  ctxt  Object  instance  toModify  JsonToken  t  Method  _getter  long  serialVersionUID  Exception  e  JsonParser  jp  
[BugLab_Argument_Swapping]^toModify = instance.invoke ( _getter ) ;^103^^^^^91^117^toModify = _getter.invoke ( instance ) ;^[CLASS] SetterlessProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] boolean  AnnotatedMethod  _annotated  method  DeserializationContext  ctxt  Object  instance  toModify  JsonToken  t  Method  _getter  long  serialVersionUID  Exception  e  JsonParser  jp  
[BugLab_Variable_Misuse]^if  ( instance == null )  {^113^^^^^91^117^if  ( toModify == null )  {^[CLASS] SetterlessProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] boolean  AnnotatedMethod  _annotated  method  DeserializationContext  ctxt  Object  instance  toModify  JsonToken  t  Method  _getter  long  serialVersionUID  Exception  e  JsonParser  jp  
[BugLab_Wrong_Operator]^if  ( toModify != null )  {^113^^^^^91^117^if  ( toModify == null )  {^[CLASS] SetterlessProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] boolean  AnnotatedMethod  _annotated  method  DeserializationContext  ctxt  Object  instance  toModify  JsonToken  t  Method  _getter  long  serialVersionUID  Exception  e  JsonParser  jp  
[BugLab_Variable_Misuse]^_valueDeserializer.deserialize ( jp, ctxt, instance ) ;^116^^^^^91^117^_valueDeserializer.deserialize ( jp, ctxt, toModify ) ;^[CLASS] SetterlessProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] boolean  AnnotatedMethod  _annotated  method  DeserializationContext  ctxt  Object  instance  toModify  JsonToken  t  Method  _getter  long  serialVersionUID  Exception  e  JsonParser  jp  
[BugLab_Argument_Swapping]^_valueDeserializer.deserialize ( ctxt, jp, toModify ) ;^116^^^^^91^117^_valueDeserializer.deserialize ( jp, ctxt, toModify ) ;^[CLASS] SetterlessProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] boolean  AnnotatedMethod  _annotated  method  DeserializationContext  ctxt  Object  instance  toModify  JsonToken  t  Method  _getter  long  serialVersionUID  Exception  e  JsonParser  jp  
[BugLab_Argument_Swapping]^_valueDeserializer.deserialize ( toModify, ctxt, jp ) ;^116^^^^^91^117^_valueDeserializer.deserialize ( jp, ctxt, toModify ) ;^[CLASS] SetterlessProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] boolean  AnnotatedMethod  _annotated  method  DeserializationContext  ctxt  Object  instance  toModify  JsonToken  t  Method  _getter  long  serialVersionUID  Exception  e  JsonParser  jp  
[BugLab_Argument_Swapping]^deserializeAndSet ( instance, ctxt, jp ) ;^124^^^^^120^126^deserializeAndSet ( jp, ctxt, instance ) ;^[CLASS] SetterlessProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] boolean  AnnotatedMethod  _annotated  method  DeserializationContext  ctxt  Object  instance  Method  _getter  long  serialVersionUID  JsonParser  jp  
[BugLab_Argument_Swapping]^deserializeAndSet ( ctxt, jp, instance ) ;^124^^^^^120^126^deserializeAndSet ( jp, ctxt, instance ) ;^[CLASS] SetterlessProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] boolean  AnnotatedMethod  _annotated  method  DeserializationContext  ctxt  Object  instance  Method  _getter  long  serialVersionUID  JsonParser  jp  
[BugLab_Argument_Swapping]^set ( value, instance ) ;^139^^^^^136^141^set ( instance, value ) ;^[CLASS] SetterlessProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  Method  _getter  boolean  AnnotatedMethod  _annotated  method  long  serialVersionUID  
