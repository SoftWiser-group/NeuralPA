[BugLab_Argument_Swapping]^return getConfig (  ) .constructSpecializedType ( subclass, baseType ) ;^100^^^^^99^101^return getConfig (  ) .constructSpecializedType ( baseType, subclass ) ;^[CLASS] DatabindContext  [METHOD] constructSpecializedType [RETURN_TYPE] JavaType   JavaType baseType Class<?> subclass [VARIABLES] JavaType  baseType  boolean  Class  subclass  
[BugLab_Argument_Swapping]^ObjectIdGenerator<?> gen =  ( implClass == null )  ? null : hi.objectIdGeneratorInstance ( config, annotated, hi ) ;^118^^^^^111^124^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( config, annotated, implClass ) ;^[CLASS] DatabindContext  [METHOD] objectIdGeneratorInstance [RETURN_TYPE] ObjectIdGenerator   Annotated annotated ObjectIdInfo objectIdInfo [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  ObjectIdGenerator  gen  boolean  MapperConfig  config  Class  implClass  ObjectIdInfo  objectIdInfo  
[BugLab_Argument_Swapping]^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( implClass, annotated, config ) ;^118^^^^^111^124^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( config, annotated, implClass ) ;^[CLASS] DatabindContext  [METHOD] objectIdGeneratorInstance [RETURN_TYPE] ObjectIdGenerator   Annotated annotated ObjectIdInfo objectIdInfo [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  ObjectIdGenerator  gen  boolean  MapperConfig  config  Class  implClass  ObjectIdInfo  objectIdInfo  
[BugLab_Argument_Swapping]^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( config, implClass, annotated ) ;^118^^^^^111^124^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( config, annotated, implClass ) ;^[CLASS] DatabindContext  [METHOD] objectIdGeneratorInstance [RETURN_TYPE] ObjectIdGenerator   Annotated annotated ObjectIdInfo objectIdInfo [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  ObjectIdGenerator  gen  boolean  MapperConfig  config  Class  implClass  ObjectIdInfo  objectIdInfo  
[BugLab_Wrong_Operator]^ObjectIdGenerator<?> gen =  ( hi != null )  ? null : hi.objectIdGeneratorInstance ( config, annotated, implClass ) ;^118^^^^^111^124^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( config, annotated, implClass ) ;^[CLASS] DatabindContext  [METHOD] objectIdGeneratorInstance [RETURN_TYPE] ObjectIdGenerator   Annotated annotated ObjectIdInfo objectIdInfo [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  ObjectIdGenerator  gen  boolean  MapperConfig  config  Class  implClass  ObjectIdInfo  objectIdInfo  
[BugLab_Variable_Misuse]^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( null, annotated, implClass ) ;^118^^^^^111^124^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( config, annotated, implClass ) ;^[CLASS] DatabindContext  [METHOD] objectIdGeneratorInstance [RETURN_TYPE] ObjectIdGenerator   Annotated annotated ObjectIdInfo objectIdInfo [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  ObjectIdGenerator  gen  boolean  MapperConfig  config  Class  implClass  ObjectIdInfo  objectIdInfo  
[BugLab_Variable_Misuse]^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( config, annotated, null ) ;^118^^^^^111^124^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( config, annotated, implClass ) ;^[CLASS] DatabindContext  [METHOD] objectIdGeneratorInstance [RETURN_TYPE] ObjectIdGenerator   Annotated annotated ObjectIdInfo objectIdInfo [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  ObjectIdGenerator  gen  boolean  MapperConfig  config  Class  implClass  ObjectIdInfo  objectIdInfo  
[BugLab_Argument_Swapping]^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( annotated, config, implClass ) ;^118^^^^^111^124^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( config, annotated, implClass ) ;^[CLASS] DatabindContext  [METHOD] objectIdGeneratorInstance [RETURN_TYPE] ObjectIdGenerator   Annotated annotated ObjectIdInfo objectIdInfo [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  ObjectIdGenerator  gen  boolean  MapperConfig  config  Class  implClass  ObjectIdInfo  objectIdInfo  
[BugLab_Argument_Swapping]^ObjectIdGenerator<?> gen =  ( annotated == null )  ? null : hi.objectIdGeneratorInstance ( config, hi, implClass ) ;^118^^^^^111^124^ObjectIdGenerator<?> gen =  ( hi == null )  ? null : hi.objectIdGeneratorInstance ( config, annotated, implClass ) ;^[CLASS] DatabindContext  [METHOD] objectIdGeneratorInstance [RETURN_TYPE] ObjectIdGenerator   Annotated annotated ObjectIdInfo objectIdInfo [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  ObjectIdGenerator  gen  boolean  MapperConfig  config  Class  implClass  ObjectIdInfo  objectIdInfo  
[BugLab_Wrong_Operator]^if  ( gen != null )  {^119^^^^^111^124^if  ( gen == null )  {^[CLASS] DatabindContext  [METHOD] objectIdGeneratorInstance [RETURN_TYPE] ObjectIdGenerator   Annotated annotated ObjectIdInfo objectIdInfo [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  ObjectIdGenerator  gen  boolean  MapperConfig  config  Class  implClass  ObjectIdInfo  objectIdInfo  
[BugLab_Argument_Swapping]^gen =  ( ObjectIdGenerator<?> )  ClassUtil.createInstance ( config, implClass.canOverrideAccessModifiers (  )  ) ;^120^121^^^^111^124^gen =  ( ObjectIdGenerator<?> )  ClassUtil.createInstance ( implClass, config.canOverrideAccessModifiers (  )  ) ;^[CLASS] DatabindContext  [METHOD] objectIdGeneratorInstance [RETURN_TYPE] ObjectIdGenerator   Annotated annotated ObjectIdInfo objectIdInfo [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  ObjectIdGenerator  gen  boolean  MapperConfig  config  Class  implClass  ObjectIdInfo  objectIdInfo  
[BugLab_Variable_Misuse]^gen =  ( ObjectIdGenerator<?> )  ClassUtil.createInstance ( 1, config.canOverrideAccessModifiers (  )  ) ;^120^121^^^^111^124^gen =  ( ObjectIdGenerator<?> )  ClassUtil.createInstance ( implClass, config.canOverrideAccessModifiers (  )  ) ;^[CLASS] DatabindContext  [METHOD] objectIdGeneratorInstance [RETURN_TYPE] ObjectIdGenerator   Annotated annotated ObjectIdInfo objectIdInfo [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  ObjectIdGenerator  gen  boolean  MapperConfig  config  Class  implClass  ObjectIdInfo  objectIdInfo  
[BugLab_Argument_Swapping]^return objectIdInfo.forScope ( gen.getScope (  )  ) ;^123^^^^^111^124^return gen.forScope ( objectIdInfo.getScope (  )  ) ;^[CLASS] DatabindContext  [METHOD] objectIdGeneratorInstance [RETURN_TYPE] ObjectIdGenerator   Annotated annotated ObjectIdInfo objectIdInfo [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  ObjectIdGenerator  gen  boolean  MapperConfig  config  Class  implClass  ObjectIdInfo  objectIdInfo  
[BugLab_Wrong_Operator]^if  ( converterDef != null )  {^137^^^^^136^164^if  ( converterDef == null )  {^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^if  ( converterDef  &  Converter<?,?> )  {^140^^^^^136^164^if  ( converterDef instanceof Converter<?,?> )  {^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^if  ( ! ( converterDef  &&  Class )  )  {^143^^^^^136^164^if  ( ! ( converterDef instanceof Class )  )  {^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^throw new IllegalStateException ( "AnnotationIntrospector returned Converter definition of type " +converterDef.getClass (  <  ) .getName (  ) +"; expected type Converter or Class<Converter> instead" ) ;^144^145^^^^136^164^throw new IllegalStateException ( "AnnotationIntrospector returned Converter definition of type " +converterDef.getClass (  ) .getName (  ) +"; expected type Converter or Class<Converter> instead" ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^throw new IllegalStateException ( "AnnotationIntrospector returned Converter definition of type " +converterDef.getClass (  ||  ) .getName (  ) +"; expected type Converter or Class<Converter> instead" ) ;^144^145^^^^136^164^throw new IllegalStateException ( "AnnotationIntrospector returned Converter definition of type " +converterDef.getClass (  ) .getName (  ) +"; expected type Converter or Class<Converter> instead" ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^throw new IllegalStateException ( "AnnotationIntrospector returned Converter definition of type " +converterDef.getClass (  |  ) .getName (  ) +"; expected type Converter or Class<Converter> instead" ) ;^144^145^^^^136^164^throw new IllegalStateException ( "AnnotationIntrospector returned Converter definition of type " +converterDef.getClass (  ) .getName (  ) +"; expected type Converter or Class<Converter> instead" ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^throw new IllegalStateException ( "AnnotationIntrospector returned Converter definition of type " +converterDef.getClass (  &&  ) .getName (  ) +"; expected type Converter or Class<Converter> instead" ) ;^144^145^^^^136^164^throw new IllegalStateException ( "AnnotationIntrospector returned Converter definition of type " +converterDef.getClass (  ) .getName (  ) +"; expected type Converter or Class<Converter> instead" ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^throw new IllegalStateException ( "AnnotationIntrospector returned Converter definition of type " +converterDef.getClass (  ^  ) .getName (  ) +"; expected type Converter or Class<Converter> instead" ) ;^144^145^^^^136^164^throw new IllegalStateException ( "AnnotationIntrospector returned Converter definition of type " +converterDef.getClass (  ) .getName (  ) +"; expected type Converter or Class<Converter> instead" ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Variable_Misuse]^if  ( 1 == Converter.None.class || converterClass == NoClass.class )  {^149^^^^^136^164^if  ( converterClass == Converter.None.class || converterClass == NoClass.class )  {^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Variable_Misuse]^if  ( converterClass == Converter.None.converterClass || converterClass == NoClass.class )  {^149^^^^^136^164^if  ( converterClass == Converter.None.class || converterClass == NoClass.class )  {^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^if  ( converterClass == Converter.None.class && converterClass == NoClass.class )  {^149^^^^^136^164^if  ( converterClass == Converter.None.class || converterClass == NoClass.class )  {^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^if  ( converterClass != Converter.None.class || converterClass == NoClass.class )  {^149^^^^^136^164^if  ( converterClass == Converter.None.class || converterClass == NoClass.class )  {^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^throw new IllegalStateException ( "AnnotationIntrospector returned Class " +converterClass.getName (  <<  ) +"; expected Class<Converter>" ) ;^153^154^^^^136^164^throw new IllegalStateException ( "AnnotationIntrospector returned Class " +converterClass.getName (  ) +"; expected Class<Converter>" ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^throw new IllegalStateException ( "AnnotationIntrospector returned Class " +converterClass.getName (  ^  ) +"; expected Class<Converter>" ) ;^153^154^^^^136^164^throw new IllegalStateException ( "AnnotationIntrospector returned Class " +converterClass.getName (  ) +"; expected Class<Converter>" ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^throw new IllegalStateException ( "AnnotationIntrospector returned Class " +converterClass.getName (   instanceof   ) +"; expected Class<Converter>" ) ;^153^154^^^^136^164^throw new IllegalStateException ( "AnnotationIntrospector returned Class " +converterClass.getName (  ) +"; expected Class<Converter>" ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^throw new IllegalStateException ( "AnnotationIntrospector returned Class " +converterClass.getName (  >>  ) +"; expected Class<Converter>" ) ;^153^154^^^^136^164^throw new IllegalStateException ( "AnnotationIntrospector returned Class " +converterClass.getName (  ) +"; expected Class<Converter>" ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^throw new IllegalStateException ( "AnnotationIntrospector returned Class " +converterClass.getName (  <  ) +"; expected Class<Converter>" ) ;^153^154^^^^136^164^throw new IllegalStateException ( "AnnotationIntrospector returned Class " +converterClass.getName (  ) +"; expected Class<Converter>" ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Argument_Swapping]^Converter<?,?> conv =  ( annotated == null )  ? null : hi.converterInstance ( config, hi, converterClass ) ;^158^^^^^136^164^Converter<?,?> conv =  ( hi == null )  ? null : hi.converterInstance ( config, annotated, converterClass ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Argument_Swapping]^Converter<?,?> conv =  ( hi == null )  ? null : hi.converterInstance ( converterClass, annotated, config ) ;^158^^^^^136^164^Converter<?,?> conv =  ( hi == null )  ? null : hi.converterInstance ( config, annotated, converterClass ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Argument_Swapping]^Converter<?,?> conv =  ( hi == null )  ? null : hi.converterInstance ( config, converterClass, annotated ) ;^158^^^^^136^164^Converter<?,?> conv =  ( hi == null )  ? null : hi.converterInstance ( config, annotated, converterClass ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^Converter<?,?> conv =  ( hi != null )  ? null : hi.converterInstance ( config, annotated, converterClass ) ;^158^^^^^136^164^Converter<?,?> conv =  ( hi == null )  ? null : hi.converterInstance ( config, annotated, converterClass ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Variable_Misuse]^Converter<?,?> conv =  ( hi == null )  ? null : hi.converterInstance ( 2, annotated, converterClass ) ;^158^^^^^136^164^Converter<?,?> conv =  ( hi == null )  ? null : hi.converterInstance ( config, annotated, converterClass ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Argument_Swapping]^Converter<?,?> conv =  ( converterClass == null )  ? null : hi.converterInstance ( config, annotated, hi ) ;^158^^^^^136^164^Converter<?,?> conv =  ( hi == null )  ? null : hi.converterInstance ( config, annotated, converterClass ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Argument_Swapping]^Converter<?,?> conv =  ( hi == null )  ? null : hi.converterInstance ( annotated, config, converterClass ) ;^158^^^^^136^164^Converter<?,?> conv =  ( hi == null )  ? null : hi.converterInstance ( config, annotated, converterClass ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Wrong_Operator]^if  ( conv != null )  {^159^^^^^136^164^if  ( conv == null )  {^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  
[BugLab_Argument_Swapping]^conv =  ( Converter<?,?> )  ClassUtil.createInstance ( config, converterClass.canOverrideAccessModifiers (  )  ) ;^160^161^^^^136^164^conv =  ( Converter<?,?> )  ClassUtil.createInstance ( converterClass, config.canOverrideAccessModifiers (  )  ) ;^[CLASS] DatabindContext  [METHOD] converterInstance [RETURN_TYPE] Converter   Annotated annotated Object converterDef [VARIABLES] HandlerInstantiator  hi  Annotated  annotated  boolean  MapperConfig  config  Converter  conv  Object  converterDef  Class  converterClass  