[P8_Replace_Mix]^private static  MethodWrapperKey KEY_FACTORY = ( MethodWrapperKey ) KeyFactory.create ( MethodWrapperKey.class ) ;^22^23^^^^22^23^private static final MethodWrapperKey KEY_FACTORY = ( MethodWrapperKey ) KeyFactory.create ( MethodWrapperKey.class ) ;^[CLASS] MethodWrapper   [VARIABLES] 
[P5_Replace_Variable]^return KEY_FACTORY.newInstance ( method.getName (  ) , ReflectUtils.getNames ( method.getParameterTypes (  )  ) .getReturnType (  ) .getName (  )  ) ;^34^35^36^^^33^37^return KEY_FACTORY.newInstance ( method.getName (  ) , ReflectUtils.getNames ( method.getParameterTypes (  )  ) , method.getReturnType (  ) .getName (  )  ) ;^[CLASS] MethodWrapper  [METHOD] create [RETURN_TYPE] Object   Method method [VARIABLES] MethodWrapperKey  KEY_FACTORY  Method  method  boolean  
[P5_Replace_Variable]^return method.newInstance ( KEY_FACTORY.getName (  ) , ReflectUtils.getNames ( method.getParameterTypes (  )  ) , method.getReturnType (  ) .getName (  )  ) ;^34^35^36^^^33^37^return KEY_FACTORY.newInstance ( method.getName (  ) , ReflectUtils.getNames ( method.getParameterTypes (  )  ) , method.getReturnType (  ) .getName (  )  ) ;^[CLASS] MethodWrapper  [METHOD] create [RETURN_TYPE] Object   Method method [VARIABLES] MethodWrapperKey  KEY_FACTORY  Method  method  boolean  
[P7_Replace_Invocation]^return KEY_FACTORY.newInstance ( method .getReturnType (  )  , ReflectUtils^34^35^36^^^33^37^return KEY_FACTORY.newInstance ( method.getName (  ) , ReflectUtils.getNames ( method.getParameterTypes (  )  ) , method.getReturnType (  ) .getName (  )  ) ;^[CLASS] MethodWrapper  [METHOD] create [RETURN_TYPE] Object   Method method [VARIABLES] MethodWrapperKey  KEY_FACTORY  Method  method  boolean  
[P7_Replace_Invocation]^return KEY_FACTORY.newInstance ( method.getName (  ) , ReflectUtils.getNames ( method.getParameterTypes (  )  ) , method .getParameterTypes (  )  .getName (  )  ) ;^34^35^36^^^33^37^return KEY_FACTORY.newInstance ( method.getName (  ) , ReflectUtils.getNames ( method.getParameterTypes (  )  ) , method.getReturnType (  ) .getName (  )  ) ;^[CLASS] MethodWrapper  [METHOD] create [RETURN_TYPE] Object   Method method [VARIABLES] MethodWrapperKey  KEY_FACTORY  Method  method  boolean  
[P14_Delete_Statement]^^34^35^36^^^33^37^return KEY_FACTORY.newInstance ( method.getName (  ) , ReflectUtils.getNames ( method.getParameterTypes (  )  ) , method.getReturnType (  ) .getName (  )  ) ;^[CLASS] MethodWrapper  [METHOD] create [RETURN_TYPE] Object   Method method [VARIABLES] MethodWrapperKey  KEY_FACTORY  Method  method  boolean  
[P8_Replace_Mix]^ReflectUtils.getNames ( method .getReturnType (  )   ) , method.getReturnType (  ) .getName (  )  ) ;^35^36^^^^33^37^ReflectUtils.getNames ( method.getParameterTypes (  )  ) , method.getReturnType (  ) .getName (  )  ) ;^[CLASS] MethodWrapper  [METHOD] create [RETURN_TYPE] Object   Method method [VARIABLES] MethodWrapperKey  KEY_FACTORY  Method  method  boolean  
[P14_Delete_Statement]^^35^36^^^^33^37^ReflectUtils.getNames ( method.getParameterTypes (  )  ) , method.getReturnType (  ) .getName (  )  ) ;^[CLASS] MethodWrapper  [METHOD] create [RETURN_TYPE] Object   Method method [VARIABLES] MethodWrapperKey  KEY_FACTORY  Method  method  boolean  
[P8_Replace_Mix]^method.getReturnType (  )  .getReturnType (  )   ) ;^36^^^^^33^37^method.getReturnType (  ) .getName (  )  ) ;^[CLASS] MethodWrapper  [METHOD] create [RETURN_TYPE] Object   Method method [VARIABLES] MethodWrapperKey  KEY_FACTORY  Method  method  boolean  
[P14_Delete_Statement]^^36^^^^^33^37^method.getReturnType (  ) .getName (  )  ) ;^[CLASS] MethodWrapper  [METHOD] create [RETURN_TYPE] Object   Method method [VARIABLES] MethodWrapperKey  KEY_FACTORY  Method  method  boolean  
[P1_Replace_Type]^List  set = new Hash List  (  ) ;^40^^^^^39^45^Set set = new HashSet (  ) ;^[CLASS] MethodWrapper  [METHOD] createSet [RETURN_TYPE] Set   Collection methods [VARIABLES] MethodWrapperKey  KEY_FACTORY  Iterator  it  Collection  methods  Set  set  boolean  
[P8_Replace_Mix]^for  ( Iterator it = methods.iterator (  ) ; it .next (  )  ; )  {^41^^^^^39^45^for  ( Iterator it = methods.iterator (  ) ; it.hasNext (  ) ; )  {^[CLASS] MethodWrapper  [METHOD] createSet [RETURN_TYPE] Set   Collection methods [VARIABLES] MethodWrapperKey  KEY_FACTORY  Iterator  it  Collection  methods  Set  set  boolean  
[P7_Replace_Invocation]^set.add ( createSet (  ( Method ) it.next (  )  )  ) ;^42^^^^^39^45^set.add ( create (  ( Method ) it.next (  )  )  ) ;^[CLASS] MethodWrapper  [METHOD] createSet [RETURN_TYPE] Set   Collection methods [VARIABLES] MethodWrapperKey  KEY_FACTORY  Iterator  it  Collection  methods  Set  set  boolean  
[P7_Replace_Invocation]^set.add ( create (  ( Method ) it .hasNext (  )   )  ) ;^42^^^^^39^45^set.add ( create (  ( Method ) it.next (  )  )  ) ;^[CLASS] MethodWrapper  [METHOD] createSet [RETURN_TYPE] Set   Collection methods [VARIABLES] MethodWrapperKey  KEY_FACTORY  Iterator  it  Collection  methods  Set  set  boolean  
[P14_Delete_Statement]^^42^^^^^39^45^set.add ( create (  ( Method ) it.next (  )  )  ) ;^[CLASS] MethodWrapper  [METHOD] createSet [RETURN_TYPE] Set   Collection methods [VARIABLES] MethodWrapperKey  KEY_FACTORY  Iterator  it  Collection  methods  Set  set  boolean  
[P14_Delete_Statement]^^41^42^43^^^39^45^for  ( Iterator it = methods.iterator (  ) ; it.hasNext (  ) ; )  { set.add ( create (  ( Method ) it.next (  )  )  ) ; }^[CLASS] MethodWrapper  [METHOD] createSet [RETURN_TYPE] Set   Collection methods [VARIABLES] MethodWrapperKey  KEY_FACTORY  Iterator  it  Collection  methods  Set  set  boolean  
