[BugLab_Variable_Misuse]^super ( EnumSet._enumClass ) ;^40^^^^^38^44^super ( EnumSet.class ) ;^[CLASS] EnumSetDeserializer  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   JavaType enumType JsonDeserializer<?> deser [VARIABLES] JavaType  _enumType  enumType  Class  _enumClass  boolean  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  
[BugLab_Variable_Misuse]^_enumType = _enumType;^41^^^^^38^44^_enumType = enumType;^[CLASS] EnumSetDeserializer  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   JavaType enumType JsonDeserializer<?> deser [VARIABLES] JavaType  _enumType  enumType  Class  _enumClass  boolean  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  
[BugLab_Variable_Misuse]^_enumClass =  ( Class<Enum> )  _enumType.getRawClass (  ) ;^42^^^^^38^44^_enumClass =  ( Class<Enum> )  enumType.getRawClass (  ) ;^[CLASS] EnumSetDeserializer  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   JavaType enumType JsonDeserializer<?> deser [VARIABLES] JavaType  _enumType  enumType  Class  _enumClass  boolean  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( deser == _enumDeserializer )  {^47^^^^^46^51^if  ( _enumDeserializer == deser )  {^[CLASS] EnumSetDeserializer  [METHOD] withDeserializer [RETURN_TYPE] EnumSetDeserializer   JsonDeserializer<?> deser [VARIABLES] JavaType  _enumType  enumType  Class  _enumClass  boolean  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( _enumDeserializer > deser )  {^47^^^^^46^51^if  ( _enumDeserializer == deser )  {^[CLASS] EnumSetDeserializer  [METHOD] withDeserializer [RETURN_TYPE] EnumSetDeserializer   JsonDeserializer<?> deser [VARIABLES] JavaType  _enumType  enumType  Class  _enumClass  boolean  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  
[BugLab_Variable_Misuse]^return new EnumSetDeserializer ( enumType, deser ) ;^50^^^^^46^51^return new EnumSetDeserializer ( _enumType, deser ) ;^[CLASS] EnumSetDeserializer  [METHOD] withDeserializer [RETURN_TYPE] EnumSetDeserializer   JsonDeserializer<?> deser [VARIABLES] JavaType  _enumType  enumType  Class  _enumClass  boolean  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  
[BugLab_Argument_Swapping]^return new EnumSetDeserializer ( deser, _enumType ) ;^50^^^^^46^51^return new EnumSetDeserializer ( _enumType, deser ) ;^[CLASS] EnumSetDeserializer  [METHOD] withDeserializer [RETURN_TYPE] EnumSetDeserializer   JsonDeserializer<?> deser [VARIABLES] JavaType  _enumType  enumType  Class  _enumClass  boolean  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  
[BugLab_Variable_Misuse]^return new EnumSetDeserializer ( _enumType, this ) ;^50^^^^^46^51^return new EnumSetDeserializer ( _enumType, deser ) ;^[CLASS] EnumSetDeserializer  [METHOD] withDeserializer [RETURN_TYPE] EnumSetDeserializer   JsonDeserializer<?> deser [VARIABLES] JavaType  _enumType  enumType  Class  _enumClass  boolean  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  
[BugLab_Wrong_Literal]^public boolean isCachable (  )  { return false; }^58^^^^^53^63^public boolean isCachable (  )  { return true; }^[CLASS] EnumSetDeserializer  [METHOD] isCachable [RETURN_TYPE] boolean   [VARIABLES] JavaType  _enumType  enumType  Class  _enumClass  boolean  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( this == null )  {^65^^^^^61^73^if  ( deser == null )  {^[CLASS] EnumSetDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] JavaType  _enumType  enumType  boolean  DeserializationContext  ctxt  Class  _enumClass  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  BeanProperty  property  
[BugLab_Wrong_Operator]^if  ( deser != null )  {^65^^^^^61^73^if  ( deser == null )  {^[CLASS] EnumSetDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] JavaType  _enumType  enumType  boolean  DeserializationContext  ctxt  Class  _enumClass  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  BeanProperty  property  
[BugLab_Wrong_Operator]^if  ( deser  <<  ContextualDeserializer )  {^68^^^^^61^73^if  ( deser instanceof ContextualDeserializer )  {^[CLASS] EnumSetDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] JavaType  _enumType  enumType  boolean  DeserializationContext  ctxt  Class  _enumClass  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  BeanProperty  property  
[BugLab_Argument_Swapping]^deser =  (  ( ContextualDeserializer )  deser ) .createContextual ( property, ctxt ) ;^69^^^^^61^73^deser =  (  ( ContextualDeserializer )  deser ) .createContextual ( ctxt, property ) ;^[CLASS] EnumSetDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] JavaType  _enumType  enumType  boolean  DeserializationContext  ctxt  Class  _enumClass  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  BeanProperty  property  
[BugLab_Variable_Misuse]^deser = ctxt.findContextualValueDeserializer ( enumType, property ) ;^66^^^^^61^73^deser = ctxt.findContextualValueDeserializer ( _enumType, property ) ;^[CLASS] EnumSetDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] JavaType  _enumType  enumType  boolean  DeserializationContext  ctxt  Class  _enumClass  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  BeanProperty  property  
[BugLab_Argument_Swapping]^deser = _enumType.findContextualValueDeserializer ( ctxt, property ) ;^66^^^^^61^73^deser = ctxt.findContextualValueDeserializer ( _enumType, property ) ;^[CLASS] EnumSetDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] JavaType  _enumType  enumType  boolean  DeserializationContext  ctxt  Class  _enumClass  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  BeanProperty  property  
[BugLab_Argument_Swapping]^deser = ctxt.findContextualValueDeserializer ( property, _enumType ) ;^66^^^^^61^73^deser = ctxt.findContextualValueDeserializer ( _enumType, property ) ;^[CLASS] EnumSetDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] JavaType  _enumType  enumType  boolean  DeserializationContext  ctxt  Class  _enumClass  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  BeanProperty  property  
[BugLab_Argument_Swapping]^deser = property.findContextualValueDeserializer ( _enumType, ctxt ) ;^66^^^^^61^73^deser = ctxt.findContextualValueDeserializer ( _enumType, property ) ;^[CLASS] EnumSetDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] JavaType  _enumType  enumType  boolean  DeserializationContext  ctxt  Class  _enumClass  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  BeanProperty  property  
[BugLab_Wrong_Operator]^if  ( deser  !=  ContextualDeserializer )  {^68^^^^^61^73^if  ( deser instanceof ContextualDeserializer )  {^[CLASS] EnumSetDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] JavaType  _enumType  enumType  boolean  DeserializationContext  ctxt  Class  _enumClass  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  BeanProperty  property  
[BugLab_Variable_Misuse]^return withDeserializer ( 0 ) ;^72^^^^^61^73^return withDeserializer ( deser ) ;^[CLASS] EnumSetDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] JavaType  _enumType  enumType  boolean  DeserializationContext  ctxt  Class  _enumClass  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  BeanProperty  property  
[BugLab_Argument_Swapping]^while  (  ( JsonToken.END_ARRAY = jp.nextToken (  )  )  != t )  {^93^^^^^85^111^while  (  ( t = jp.nextToken (  )  )  != JsonToken.END_ARRAY )  {^[CLASS] EnumSetDeserializer  [METHOD] deserialize [RETURN_TYPE] EnumSet   JsonParser jp DeserializationContext ctxt [VARIABLES] Enum  value  JavaType  _enumType  enumType  boolean  EnumSet  result  DeserializationContext  ctxt  Class  _enumClass  JsonToken  t  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  JsonParser  jp  
[BugLab_Wrong_Operator]^while  (  ( t = jp.nextToken (  )  )  >= JsonToken.END_ARRAY )  {^93^^^^^85^111^while  (  ( t = jp.nextToken (  )  )  != JsonToken.END_ARRAY )  {^[CLASS] EnumSetDeserializer  [METHOD] deserialize [RETURN_TYPE] EnumSet   JsonParser jp DeserializationContext ctxt [VARIABLES] Enum  value  JavaType  _enumType  enumType  boolean  EnumSet  result  DeserializationContext  ctxt  Class  _enumClass  JsonToken  t  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  JsonParser  jp  
[BugLab_Wrong_Operator]^if  ( t >= JsonToken.VALUE_NULL )  {^99^^^^^85^111^if  ( t == JsonToken.VALUE_NULL )  {^[CLASS] EnumSetDeserializer  [METHOD] deserialize [RETURN_TYPE] EnumSet   JsonParser jp DeserializationContext ctxt [VARIABLES] Enum  value  JavaType  _enumType  enumType  boolean  EnumSet  result  DeserializationContext  ctxt  Class  _enumClass  JsonToken  t  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  JsonParser  jp  
[BugLab_Wrong_Operator]^if  ( value == null )  {^106^^^^^85^111^if  ( value != null )  {^[CLASS] EnumSetDeserializer  [METHOD] deserialize [RETURN_TYPE] EnumSet   JsonParser jp DeserializationContext ctxt [VARIABLES] Enum  value  JavaType  _enumType  enumType  boolean  EnumSet  result  DeserializationContext  ctxt  Class  _enumClass  JsonToken  t  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  JsonParser  jp  
[BugLab_Argument_Swapping]^Enum<?> value = _enumDeserializer.deserialize ( ctxt, jp ) ;^102^^^^^85^111^Enum<?> value = _enumDeserializer.deserialize ( jp, ctxt ) ;^[CLASS] EnumSetDeserializer  [METHOD] deserialize [RETURN_TYPE] EnumSet   JsonParser jp DeserializationContext ctxt [VARIABLES] Enum  value  JavaType  _enumType  enumType  boolean  EnumSet  result  DeserializationContext  ctxt  Class  _enumClass  JsonToken  t  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  JsonParser  jp  
[BugLab_Argument_Swapping]^Enum<?> value = ctxt.deserialize ( jp, _enumDeserializer ) ;^102^^^^^85^111^Enum<?> value = _enumDeserializer.deserialize ( jp, ctxt ) ;^[CLASS] EnumSetDeserializer  [METHOD] deserialize [RETURN_TYPE] EnumSet   JsonParser jp DeserializationContext ctxt [VARIABLES] Enum  value  JavaType  _enumType  enumType  boolean  EnumSet  result  DeserializationContext  ctxt  Class  _enumClass  JsonToken  t  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  JsonParser  jp  
[BugLab_Wrong_Operator]^if  ( t != JsonToken.VALUE_NULL )  {^99^^^^^85^111^if  ( t == JsonToken.VALUE_NULL )  {^[CLASS] EnumSetDeserializer  [METHOD] deserialize [RETURN_TYPE] EnumSet   JsonParser jp DeserializationContext ctxt [VARIABLES] Enum  value  JavaType  _enumType  enumType  boolean  EnumSet  result  DeserializationContext  ctxt  Class  _enumClass  JsonToken  t  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  JsonParser  jp  
[BugLab_Argument_Swapping]^Enum<?> value = jp.deserialize ( _enumDeserializer, ctxt ) ;^102^^^^^85^111^Enum<?> value = _enumDeserializer.deserialize ( jp, ctxt ) ;^[CLASS] EnumSetDeserializer  [METHOD] deserialize [RETURN_TYPE] EnumSet   JsonParser jp DeserializationContext ctxt [VARIABLES] Enum  value  JavaType  _enumType  enumType  boolean  EnumSet  result  DeserializationContext  ctxt  Class  _enumClass  JsonToken  t  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  JsonParser  jp  
[BugLab_Variable_Misuse]^Enum<?> value = null.deserialize ( jp, ctxt ) ;^102^^^^^85^111^Enum<?> value = _enumDeserializer.deserialize ( jp, ctxt ) ;^[CLASS] EnumSetDeserializer  [METHOD] deserialize [RETURN_TYPE] EnumSet   JsonParser jp DeserializationContext ctxt [VARIABLES] Enum  value  JavaType  _enumType  enumType  boolean  EnumSet  result  DeserializationContext  ctxt  Class  _enumClass  JsonToken  t  JsonDeserializer  _enumDeserializer  deser  long  serialVersionUID  JsonParser  jp  
[BugLab_Argument_Swapping]^return ctxt.deserializeTypedFromArray ( jp, typeDeserializer ) ;^118^^^^^114^119^return typeDeserializer.deserializeTypedFromArray ( jp, ctxt ) ;^[CLASS] EnumSetDeserializer  [METHOD] deserializeWithType [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt TypeDeserializer typeDeserializer [VARIABLES] JavaType  _enumType  enumType  boolean  DeserializationContext  ctxt  Class  _enumClass  JsonDeserializer  _enumDeserializer  deser  TypeDeserializer  typeDeserializer  long  serialVersionUID  JsonParser  jp  
[BugLab_Argument_Swapping]^return typeDeserializer.deserializeTypedFromArray ( ctxt, jp ) ;^118^^^^^114^119^return typeDeserializer.deserializeTypedFromArray ( jp, ctxt ) ;^[CLASS] EnumSetDeserializer  [METHOD] deserializeWithType [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt TypeDeserializer typeDeserializer [VARIABLES] JavaType  _enumType  enumType  boolean  DeserializationContext  ctxt  Class  _enumClass  JsonDeserializer  _enumDeserializer  deser  TypeDeserializer  typeDeserializer  long  serialVersionUID  JsonParser  jp  
