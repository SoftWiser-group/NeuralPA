[BugLab_Wrong_Literal]^public static final DispatcherGenerator INSTANCE = new DispatcherGenerator ( true ) ;^24^25^^^^24^25^public static final DispatcherGenerator INSTANCE = new DispatcherGenerator ( false ) ;^[CLASS] DispatcherGenerator   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final DispatcherGenerator PROXY_REF_INSTANCE = new DispatcherGenerator ( false ) ;^26^27^^^^26^27^public static final DispatcherGenerator PROXY_REF_INSTANCE = new DispatcherGenerator ( true ) ;^[CLASS] DispatcherGenerator   [VARIABLES] 
[BugLab_Variable_Misuse]^e.invoke_interface ( PROXY_REF_DISPATCHER, LOAD_OBJECT ) ;^54^^^^^44^63^e.invoke_interface ( DISPATCHER, LOAD_OBJECT ) ;^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
[BugLab_Variable_Misuse]^e.invoke_interface ( DISPATCHER, PROXY_REF_LOAD_OBJECT ) ;^54^^^^^44^63^e.invoke_interface ( DISPATCHER, LOAD_OBJECT ) ;^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
[BugLab_Argument_Swapping]^e.invoke_interface ( LOAD_OBJECT, DISPATCHER ) ;^54^^^^^44^63^e.invoke_interface ( DISPATCHER, LOAD_OBJECT ) ;^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
[BugLab_Variable_Misuse]^e.invoke_interface ( DISPATCHER, PROXY_REF_LOAD_OBJECT ) ;^52^^^^^44^63^e.invoke_interface ( PROXY_REF_DISPATCHER, PROXY_REF_LOAD_OBJECT ) ;^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
[BugLab_Variable_Misuse]^e.invoke_interface ( PROXY_REF_DISPATCHER, LOAD_OBJECT ) ;^52^^^^^44^63^e.invoke_interface ( PROXY_REF_DISPATCHER, PROXY_REF_LOAD_OBJECT ) ;^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
[BugLab_Argument_Swapping]^e.invoke_interface ( PROXY_REF_LOAD_OBJECT, PROXY_REF_DISPATCHER ) ;^52^^^^^44^63^e.invoke_interface ( PROXY_REF_DISPATCHER, PROXY_REF_LOAD_OBJECT ) ;^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
[BugLab_Argument_Swapping]^CodeEmitter e = method.beginMethod ( ce, context ) ;^48^^^^^44^63^CodeEmitter e = context.beginMethod ( ce, method ) ;^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
[BugLab_Argument_Swapping]^CodeEmitter e = ce.beginMethod ( context, method ) ;^48^^^^^44^63^CodeEmitter e = context.beginMethod ( ce, method ) ;^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
[BugLab_Argument_Swapping]^context.emitCallback ( context, e.getIndex ( method )  ) ;^49^^^^^44^63^context.emitCallback ( e, context.getIndex ( method )  ) ;^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
[BugLab_Argument_Swapping]^context.emitCallback ( method, context.getIndex ( e )  ) ;^49^^^^^44^63^context.emitCallback ( e, context.getIndex ( method )  ) ;^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
[BugLab_Argument_Swapping]^context.emitCallback ( e, method.getIndex ( context )  ) ;^49^^^^^44^63^context.emitCallback ( e, context.getIndex ( method )  ) ;^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
[BugLab_Argument_Swapping]^for  ( Iterator it = method.iterator (  ) ; it.hasNext (  ) ; )  {^45^^^^^44^63^for  ( Iterator it = methods.iterator (  ) ; it.hasNext (  ) ; )  {^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
[BugLab_Argument_Swapping]^CodeEmitter e = context.beginMethod ( method, ce ) ;^48^^^^^44^63^CodeEmitter e = context.beginMethod ( ce, method ) ;^[CLASS] DispatcherGenerator  [METHOD] generate [RETURN_TYPE] void   ClassEmitter ce Context context List methods [VARIABLES] DispatcherGenerator  INSTANCE  PROXY_REF_INSTANCE  Context  context  CodeEmitter  e  Type  DISPATCHER  PROXY_REF_DISPATCHER  boolean  proxyRef  Signature  LOAD_OBJECT  PROXY_REF_LOAD_OBJECT  ClassEmitter  ce  Iterator  it  List  methods  MethodInfo  method  
