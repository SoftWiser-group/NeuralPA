[BugLab_Variable_Misuse]^_xmlTextProperty = _xmlTextProperty;^47^^^^^44^50^_xmlTextProperty = prop;^[CLASS] XmlTextDeserializer  [METHOD] <init> [RETURN_TYPE] SettableBeanProperty)   BeanDeserializerBase delegate SettableBeanProperty prop [VARIABLES] BeanDeserializerBase  delegate  boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  long  serialVersionUID  int  _xmlTextPropertyIndex  
[BugLab_Variable_Misuse]^_xmlTextPropertyIndex = _xmlTextProperty.getPropertyIndex (  ) ;^48^^^^^44^50^_xmlTextPropertyIndex = prop.getPropertyIndex (  ) ;^[CLASS] XmlTextDeserializer  [METHOD] <init> [RETURN_TYPE] SettableBeanProperty)   BeanDeserializerBase delegate SettableBeanProperty prop [VARIABLES] BeanDeserializerBase  delegate  boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  long  serialVersionUID  int  _xmlTextPropertyIndex  
[BugLab_Variable_Misuse]^_xmlTextPropertyIndex = _xmlTextPropertyIndex;^55^^^^^52^58^_xmlTextPropertyIndex = textPropIndex;^[CLASS] XmlTextDeserializer  [METHOD] <init> [RETURN_TYPE] BeanDeserializerBase,int)   BeanDeserializerBase delegate int textPropIndex [VARIABLES] BeanDeserializerBase  delegate  boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  
[BugLab_Variable_Misuse]^_xmlTextProperty = delegate.findProperty ( _xmlTextPropertyIndex ) ;^57^^^^^52^58^_xmlTextProperty = delegate.findProperty ( textPropIndex ) ;^[CLASS] XmlTextDeserializer  [METHOD] <init> [RETURN_TYPE] BeanDeserializerBase,int)   BeanDeserializerBase delegate int textPropIndex [VARIABLES] BeanDeserializerBase  delegate  boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  
[BugLab_Argument_Swapping]^_xmlTextProperty = textPropIndex.findProperty ( delegate ) ;^57^^^^^52^58^_xmlTextProperty = delegate.findProperty ( textPropIndex ) ;^[CLASS] XmlTextDeserializer  [METHOD] <init> [RETURN_TYPE] BeanDeserializerBase,int)   BeanDeserializerBase delegate int textPropIndex [VARIABLES] BeanDeserializerBase  delegate  boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  
[BugLab_Variable_Misuse]^return new XmlTextDeserializer ( _verifyDeserType ( 3 ) , _xmlTextPropertyIndex ) ;^78^^^^^74^79^return new XmlTextDeserializer ( _verifyDeserType ( _delegatee ) , _xmlTextPropertyIndex ) ;^[CLASS] XmlTextDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  BeanProperty  property  
[BugLab_Variable_Misuse]^return new XmlTextDeserializer ( _verifyDeserType ( _delegatee ) , textPropIndex ) ;^78^^^^^74^79^return new XmlTextDeserializer ( _verifyDeserType ( _delegatee ) , _xmlTextPropertyIndex ) ;^[CLASS] XmlTextDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  BeanProperty  property  
[BugLab_Argument_Swapping]^return new XmlTextDeserializer ( _verifyDeserType ( _xmlTextPropertyIndex ) , _delegatee ) ;^78^^^^^74^79^return new XmlTextDeserializer ( _verifyDeserType ( _delegatee ) , _xmlTextPropertyIndex ) ;^[CLASS] XmlTextDeserializer  [METHOD] createContextual [RETURN_TYPE] JsonDeserializer   DeserializationContext ctxt BeanProperty property [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  BeanProperty  property  
[BugLab_Argument_Swapping]^if  ( JsonToken.VALUE_STRING.getCurrentToken (  )  == jp )  {^91^^^^^88^97^if  ( jp.getCurrentToken (  )  == JsonToken.VALUE_STRING )  {^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Wrong_Operator]^if  ( jp.getCurrentToken (  )  < JsonToken.VALUE_STRING )  {^91^^^^^88^97^if  ( jp.getCurrentToken (  )  == JsonToken.VALUE_STRING )  {^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^Object bean = ctxt.createUsingDefault ( _valueInstantiator ) ;^92^^^^^88^97^Object bean = _valueInstantiator.createUsingDefault ( ctxt ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^_xmlTextProperty.deserializeAndSet ( ctxt, jp, bean ) ;^93^^^^^88^97^_xmlTextProperty.deserializeAndSet ( jp, ctxt, bean ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^_xmlTextProperty.deserializeAndSet ( bean, ctxt, jp ) ;^93^^^^^88^97^_xmlTextProperty.deserializeAndSet ( jp, ctxt, bean ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^_xmlTextProperty.deserializeAndSet ( jp, bean, ctxt ) ;^93^^^^^88^97^_xmlTextProperty.deserializeAndSet ( jp, ctxt, bean ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^return _delegatee.deserialize ( ctxt,  jp ) ;^96^^^^^88^97^return _delegatee.deserialize ( jp,  ctxt ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^return ctxt.deserialize ( jp,  _delegatee ) ;^96^^^^^88^97^return _delegatee.deserialize ( jp,  ctxt ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^return jp.deserialize ( _delegatee,  ctxt ) ;^96^^^^^88^97^return _delegatee.deserialize ( jp,  ctxt ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Wrong_Operator]^if  ( jp.getCurrentToken (  )  != JsonToken.VALUE_STRING )  {^105^^^^^101^110^if  ( jp.getCurrentToken (  )  == JsonToken.VALUE_STRING )  {^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object bean [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^_xmlTextProperty.deserializeAndSet ( bean, ctxt, jp ) ;^106^^^^^101^110^_xmlTextProperty.deserializeAndSet ( jp, ctxt, bean ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object bean [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^_xmlTextProperty.deserializeAndSet ( jp, bean, ctxt ) ;^106^^^^^101^110^_xmlTextProperty.deserializeAndSet ( jp, ctxt, bean ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object bean [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^_xmlTextProperty.deserializeAndSet ( ctxt, jp, bean ) ;^106^^^^^101^110^_xmlTextProperty.deserializeAndSet ( jp, ctxt, bean ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object bean [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^return  (  ( JsonDeserializer<Object> ) _delegatee ) .deserialize ( bean, ctxt, jp ) ;^109^^^^^101^110^return  (  ( JsonDeserializer<Object> ) _delegatee ) .deserialize ( jp, ctxt, bean ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object bean [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^return  (  ( JsonDeserializer<Object> ) _delegatee ) .deserialize ( jp, bean, ctxt ) ;^109^^^^^101^110^return  (  ( JsonDeserializer<Object> ) _delegatee ) .deserialize ( jp, ctxt, bean ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object bean [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^return  (  ( JsonDeserializer<Object> ) _delegatee ) .deserialize ( ctxt, jp, bean ) ;^109^^^^^101^110^return  (  ( JsonDeserializer<Object> ) _delegatee ) .deserialize ( jp, ctxt, bean ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserialize [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object bean [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  Object  bean  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^return jp.deserializeWithType ( _delegatee, ctxt, typeDeserializer ) ;^117^^^^^113^118^return _delegatee.deserializeWithType ( jp, ctxt, typeDeserializer ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserializeWithType [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt TypeDeserializer typeDeserializer [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  TypeDeserializer  typeDeserializer  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^return _delegatee.deserializeWithType ( jp, typeDeserializer, ctxt ) ;^117^^^^^113^118^return _delegatee.deserializeWithType ( jp, ctxt, typeDeserializer ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserializeWithType [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt TypeDeserializer typeDeserializer [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  TypeDeserializer  typeDeserializer  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^return typeDeserializer.deserializeWithType ( jp, ctxt, _delegatee ) ;^117^^^^^113^118^return _delegatee.deserializeWithType ( jp, ctxt, typeDeserializer ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserializeWithType [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt TypeDeserializer typeDeserializer [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  TypeDeserializer  typeDeserializer  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Argument_Swapping]^return ctxt.deserializeWithType ( jp, _delegatee, typeDeserializer ) ;^117^^^^^113^118^return _delegatee.deserializeWithType ( jp, ctxt, typeDeserializer ) ;^[CLASS] XmlTextDeserializer  [METHOD] deserializeWithType [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt TypeDeserializer typeDeserializer [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  ValueInstantiator  _valueInstantiator  DeserializationContext  ctxt  TypeDeserializer  typeDeserializer  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  JsonParser  jp  
[BugLab_Variable_Misuse]^if  ( ! ( 1 instanceof BeanDeserializerBase )  )  {^128^^^^^126^133^if  ( ! ( deser instanceof BeanDeserializerBase )  )  {^[CLASS] XmlTextDeserializer  [METHOD] _verifyDeserType [RETURN_TYPE] BeanDeserializerBase   JsonDeserializer<?> deser [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  JsonDeserializer  deser  ValueInstantiator  _valueInstantiator  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  
[BugLab_Wrong_Operator]^if  ( ! ( deser  >=  BeanDeserializerBase )  )  {^128^^^^^126^133^if  ( ! ( deser instanceof BeanDeserializerBase )  )  {^[CLASS] XmlTextDeserializer  [METHOD] _verifyDeserType [RETURN_TYPE] BeanDeserializerBase   JsonDeserializer<?> deser [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  JsonDeserializer  deser  ValueInstantiator  _valueInstantiator  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Can not change delegate to be of type " +deser.getClass (  >  ) .getName (  )  ) ;^129^130^^^^126^133^throw new IllegalArgumentException ( "Can not change delegate to be of type " +deser.getClass (  ) .getName (  )  ) ;^[CLASS] XmlTextDeserializer  [METHOD] _verifyDeserType [RETURN_TYPE] BeanDeserializerBase   JsonDeserializer<?> deser [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  JsonDeserializer  deser  ValueInstantiator  _valueInstantiator  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Can not change delegate to be of type " +deser.getClass (  ==  ) .getName (  )  ) ;^129^130^^^^126^133^throw new IllegalArgumentException ( "Can not change delegate to be of type " +deser.getClass (  ) .getName (  )  ) ;^[CLASS] XmlTextDeserializer  [METHOD] _verifyDeserType [RETURN_TYPE] BeanDeserializerBase   JsonDeserializer<?> deser [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  JsonDeserializer  deser  ValueInstantiator  _valueInstantiator  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Can not change delegate to be of type " +deser.getClass (   instanceof   ) .getName (  )  ) ;^129^130^^^^126^133^throw new IllegalArgumentException ( "Can not change delegate to be of type " +deser.getClass (  ) .getName (  )  ) ;^[CLASS] XmlTextDeserializer  [METHOD] _verifyDeserType [RETURN_TYPE] BeanDeserializerBase   JsonDeserializer<?> deser [VARIABLES] boolean  SettableBeanProperty  _xmlTextProperty  prop  JsonDeserializer  deser  ValueInstantiator  _valueInstantiator  long  serialVersionUID  int  _xmlTextPropertyIndex  textPropIndex  
