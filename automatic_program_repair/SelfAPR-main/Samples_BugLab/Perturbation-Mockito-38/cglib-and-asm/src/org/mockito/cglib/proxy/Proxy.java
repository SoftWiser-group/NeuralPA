[BugLab_Wrong_Operator]^if  ( ! ( name.equals ( "hashCode" )  && name.equals ( "equals" )  || name.equals ( "toString" )  )  )  {^46^47^48^^^43^53^if  ( ! ( name.equals ( "hashCode" )  || name.equals ( "equals" )  || name.equals ( "toString" )  )  )  {^[CLASS] Proxy 1 ProxyImpl  [METHOD] accept [RETURN_TYPE] int   Method method [VARIABLES] CallbackFilter  BAD_OBJECT_METHOD_FILTER  Method  method  String  name  boolean  InvocationHandler  h  
[BugLab_Wrong_Literal]^return 2;^49^^^^^43^53^return 1;^[CLASS] Proxy 1 ProxyImpl  [METHOD] accept [RETURN_TYPE] int   Method method [VARIABLES] CallbackFilter  BAD_OBJECT_METHOD_FILTER  Method  method  String  name  boolean  InvocationHandler  h  
[BugLab_Wrong_Literal]^return 0;^49^^^^^43^53^return 1;^[CLASS] Proxy 1 ProxyImpl  [METHOD] accept [RETURN_TYPE] int   Method method [VARIABLES] CallbackFilter  BAD_OBJECT_METHOD_FILTER  Method  method  String  name  boolean  InvocationHandler  h  
[BugLab_Wrong_Operator]^if  ( ! ( proxy  &&  ProxyImpl )  )  {^69^^^^^68^73^if  ( ! ( proxy instanceof ProxyImpl )  )  {^[CLASS] Proxy 1 ProxyImpl  [METHOD] getInvocationHandler [RETURN_TYPE] InvocationHandler   Object proxy [VARIABLES] Object  proxy  CallbackFilter  BAD_OBJECT_METHOD_FILTER  boolean  InvocationHandler  h  
[BugLab_Wrong_Literal]^e.setUseFactory ( true ) ;^84^^^^^75^86^e.setUseFactory ( false ) ;^[CLASS] Proxy 1 ProxyImpl  [METHOD] getProxyClass [RETURN_TYPE] Class   ClassLoader loader Class[] interfaces [VARIABLES] ClassLoader  loader  CallbackFilter  BAD_OBJECT_METHOD_FILTER  Class[]  interfaces  boolean  Enhancer  e  InvocationHandler  h  
[BugLab_Variable_Misuse]^return clazz.getConstructor ( new Class[]{ InvocationHandler.clazz } ) .newInstance ( new Object[]{ h } ) ;^95^^^^^92^101^return clazz.getConstructor ( new Class[]{ InvocationHandler.class } ) .newInstance ( new Object[]{ h } ) ;^[CLASS] Proxy 1 ProxyImpl  [METHOD] newProxyInstance [RETURN_TYPE] Object   ClassLoader loader Class[] interfaces InvocationHandler h [VARIABLES] Class[]  interfaces  RuntimeException  e  boolean  ClassLoader  loader  Class  clazz  CallbackFilter  BAD_OBJECT_METHOD_FILTER  InvocationHandler  h  Exception  e  
[BugLab_Argument_Swapping]^Class clazz = getProxyClass ( interfaces, loader ) ;^94^^^^^92^101^Class clazz = getProxyClass ( loader, interfaces ) ;^[CLASS] Proxy 1 ProxyImpl  [METHOD] newProxyInstance [RETURN_TYPE] Object   ClassLoader loader Class[] interfaces InvocationHandler h [VARIABLES] Class[]  interfaces  RuntimeException  e  boolean  ClassLoader  loader  Class  clazz  CallbackFilter  BAD_OBJECT_METHOD_FILTER  InvocationHandler  h  Exception  e  
[BugLab_Argument_Swapping]^return h.getConstructor ( new Class[]{ InvocationHandler.class } ) .newInstance ( new Object[]{ clazz } ) ;^95^^^^^92^101^return clazz.getConstructor ( new Class[]{ InvocationHandler.class } ) .newInstance ( new Object[]{ h } ) ;^[CLASS] Proxy 1 ProxyImpl  [METHOD] newProxyInstance [RETURN_TYPE] Object   ClassLoader loader Class[] interfaces InvocationHandler h [VARIABLES] Class[]  interfaces  RuntimeException  e  boolean  ClassLoader  loader  Class  clazz  CallbackFilter  BAD_OBJECT_METHOD_FILTER  InvocationHandler  h  Exception  e  
[BugLab_Wrong_Operator]^if  ( ! ( name.equals ( "hashCode" )  && name.equals ( "equals" )  || name.equals ( "toString" )  )  )  {^46^47^48^^^43^53^if  ( ! ( name.equals ( "hashCode" )  || name.equals ( "equals" )  || name.equals ( "toString" )  )  )  {^[CLASS] 1  [METHOD] accept [RETURN_TYPE] int   Method method [VARIABLES] boolean  Method  method  String  name  
[BugLab_Wrong_Literal]^return 2;^49^^^^^43^53^return 1;^[CLASS] 1  [METHOD] accept [RETURN_TYPE] int   Method method [VARIABLES] boolean  Method  method  String  name  
[BugLab_Wrong_Literal]^return 0;^49^^^^^43^53^return 1;^[CLASS] 1  [METHOD] accept [RETURN_TYPE] int   Method method [VARIABLES] boolean  Method  method  String  name  
[BugLab_Wrong_Literal]^return 1;^52^^^^^43^53^return 0;^[CLASS] 1  [METHOD] accept [RETURN_TYPE] int   Method method [VARIABLES] boolean  Method  method  String  name  
