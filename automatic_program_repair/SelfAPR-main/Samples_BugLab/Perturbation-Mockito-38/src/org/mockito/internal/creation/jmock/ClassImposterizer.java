[BugLab_Wrong_Literal]^return method.isBridge (  )  ? 1 : -1;^48^^^^^47^49^return method.isBridge (  )  ? 1 : 0;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] accept [RETURN_TYPE] int   Method method [VARIABLES] ObjenesisStd  objenesis  CallbackFilter  IGNORE_BRIDGE_METHODS  Method  method  boolean  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  ClassImposterizer  INSTANCE  
[BugLab_Wrong_Operator]^return !type.isPrimitive (  )  || !Modifier.isFinal ( type.getModifiers (  )  )  && !type.isAnonymousClass (  ) ;^53^^^^^52^54^return !type.isPrimitive (  )  && !Modifier.isFinal ( type.getModifiers (  )  )  && !type.isAnonymousClass (  ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] canImposterise [RETURN_TYPE] boolean   Class<?> type [VARIABLES] Class  type  ObjenesisStd  objenesis  CallbackFilter  IGNORE_BRIDGE_METHODS  boolean  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  ClassImposterizer  INSTANCE  
[BugLab_Argument_Swapping]^return interceptor.cast ( createProxy ( proxyClass, mockedType )  ) ;^60^^^^^56^64^return mockedType.cast ( createProxy ( proxyClass, interceptor )  ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] imposterise [RETURN_TYPE] <T>   MethodInterceptor interceptor Class<T> mockedType  ancillaryTypes [VARIABLES] ObjenesisStd  objenesis  MethodInterceptor  interceptor  Class[]  ancillaryTypes  boolean  ClassImposterizer  INSTANCE  Class  mockedType  proxyClass  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Argument_Swapping]^return mockedType.cast ( createProxy ( interceptor, proxyClass )  ) ;^60^^^^^56^64^return mockedType.cast ( createProxy ( proxyClass, interceptor )  ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] imposterise [RETURN_TYPE] <T>   MethodInterceptor interceptor Class<T> mockedType  ancillaryTypes [VARIABLES] ObjenesisStd  objenesis  MethodInterceptor  interceptor  Class[]  ancillaryTypes  boolean  ClassImposterizer  INSTANCE  Class  mockedType  proxyClass  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Argument_Swapping]^Class<?> proxyClass = createProxyClass ( ancillaryTypes, mockedType ) ;^59^^^^^56^64^Class<?> proxyClass = createProxyClass ( mockedType, ancillaryTypes ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] imposterise [RETURN_TYPE] <T>   MethodInterceptor interceptor Class<T> mockedType  ancillaryTypes [VARIABLES] ObjenesisStd  objenesis  MethodInterceptor  interceptor  Class[]  ancillaryTypes  boolean  ClassImposterizer  INSTANCE  Class  mockedType  proxyClass  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Literal]^setConstructorsAccessible ( mockedType, false ) ;^58^^^^^56^64^setConstructorsAccessible ( mockedType, true ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] imposterise [RETURN_TYPE] <T>   MethodInterceptor interceptor Class<T> mockedType  ancillaryTypes [VARIABLES] ObjenesisStd  objenesis  MethodInterceptor  interceptor  Class[]  ancillaryTypes  boolean  ClassImposterizer  INSTANCE  Class  mockedType  proxyClass  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Argument_Swapping]^return proxyClass.cast ( createProxy ( mockedType, interceptor )  ) ;^60^^^^^56^64^return mockedType.cast ( createProxy ( proxyClass, interceptor )  ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] imposterise [RETURN_TYPE] <T>   MethodInterceptor interceptor Class<T> mockedType  ancillaryTypes [VARIABLES] ObjenesisStd  objenesis  MethodInterceptor  interceptor  Class[]  ancillaryTypes  boolean  ClassImposterizer  INSTANCE  Class  mockedType  proxyClass  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Literal]^setConstructorsAccessible ( mockedType, true ) ;^62^^^^^56^64^setConstructorsAccessible ( mockedType, false ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] imposterise [RETURN_TYPE] <T>   MethodInterceptor interceptor Class<T> mockedType  ancillaryTypes [VARIABLES] ObjenesisStd  objenesis  MethodInterceptor  interceptor  Class[]  ancillaryTypes  boolean  ClassImposterizer  INSTANCE  Class  mockedType  proxyClass  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Variable_Misuse]^if  ( mockedType == Object.sc )  {^73^^^^^58^88^if  ( mockedType == Object.class )  {^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^if  ( mockedType <= Object.class )  {^73^^^^^58^88^if  ( mockedType == Object.class )  {^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Variable_Misuse]^enhancer.setClassLoader ( SearchingClassLoader.combineLoadersOf ( this )  ) ;^84^^^^^69^99^enhancer.setClassLoader ( SearchingClassLoader.combineLoadersOf ( mockedType )  ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Literal]^enhancer.setUseFactory ( false ) ;^85^^^^^70^100^enhancer.setUseFactory ( true ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Variable_Misuse]^enhancer.setInterfaces ( null ) ;^91^^^^^86^92^enhancer.setInterfaces ( interfaces ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Argument_Swapping]^enhancer.setInterfaces ( prepend ( interfaces, mockedType )  ) ;^88^^^^^73^103^enhancer.setInterfaces ( prepend ( mockedType, interfaces )  ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Variable_Misuse]^enhancer.setInterfaces ( prepend ( 2, interfaces )  ) ;^88^^^^^73^103^enhancer.setInterfaces ( prepend ( mockedType, interfaces )  ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Variable_Misuse]^enhancer.setSupersc ( Object.class ) ;^87^^^^^72^102^enhancer.setSuperclass ( Object.class ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Variable_Misuse]^enhancer.setInterfaces ( prepend ( 0, interfaces )  ) ;^88^^^^^73^103^enhancer.setInterfaces ( prepend ( mockedType, interfaces )  ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Variable_Misuse]^enhancer.setInterfaces ( prepend ( mockedType, null )  ) ;^88^^^^^73^103^enhancer.setInterfaces ( prepend ( mockedType, interfaces )  ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^if  ( mockedType.getSigners (  )  == null )  {^95^^^^^80^110^if  ( mockedType.getSigners (  )  != null )  {^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^throw new MockitoException ( "\n"  <=  "Mockito cannot mock this class: "  <=  mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^105^106^107^108^^90^120^throw new MockitoException ( "\n" + "Mockito cannot mock this class: " + mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^throw new MockitoException ( "\n"  ==  "Mockito cannot mock this class: " + mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^105^106^107^108^^90^120^throw new MockitoException ( "\n" + "Mockito cannot mock this class: " + mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^throw new MockitoException ( "\n"  >=  "Mockito cannot mock this class: "  >=  mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^105^106^107^108^^90^120^throw new MockitoException ( "\n" + "Mockito cannot mock this class: " + mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^throw new MockitoException ( "\n"  >=  "Mockito cannot mock this class: " + mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^105^106^107^108^^90^120^throw new MockitoException ( "\n" + "Mockito cannot mock this class: " + mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^throw new MockitoException ( "\n"  <<  "Mockito cannot mock this class: " + mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^105^106^107^108^^90^120^throw new MockitoException ( "\n" + "Mockito cannot mock this class: " + mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^throw new MockitoException ( "\n"  ==  "Mockito cannot mock this class: "  ==  mockedType + ".\n" + "Mockito can only mock visible & non-final classes" ) ;^110^111^112^113^^95^125^throw new MockitoException ( "\n" + "Mockito cannot mock this class: " + mockedType + ".\n" + "Mockito can only mock visible & non-final classes" ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^throw new MockitoException ( "\n"  >=  "Mockito cannot mock this class: " + mockedType + ".\n" + "Mockito can only mock visible & non-final classes" ) ;^110^111^112^113^^95^125^throw new MockitoException ( "\n" + "Mockito cannot mock this class: " + mockedType + ".\n" + "Mockito can only mock visible & non-final classes" ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^throw new MockitoException ( "\n"  |  "Mockito cannot mock this class: "  |  mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^105^106^107^108^^90^120^throw new MockitoException ( "\n" + "Mockito cannot mock this class: " + mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^throw new MockitoException ( "\n"  &&  "Mockito cannot mock this class: " + mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^105^106^107^108^^90^120^throw new MockitoException ( "\n" + "Mockito cannot mock this class: " + mockedType + ".\n" + "Most likely it is a private class that is not visible by Mockito" ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^throw new MockitoException ( "\n"  !=  "Mockito cannot mock this class: "  !=  mockedType + ".\n" + "Mockito can only mock visible & non-final classes" ) ;^110^111^112^113^^95^125^throw new MockitoException ( "\n" + "Mockito cannot mock this class: " + mockedType + ".\n" + "Mockito can only mock visible & non-final classes" ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Operator]^throw new MockitoException ( "\n"  &&  "Mockito cannot mock this class: " + mockedType + ".\n" + "Mockito can only mock visible & non-final classes" ) ;^110^111^112^113^^95^125^throw new MockitoException ( "\n" + "Mockito cannot mock this class: " + mockedType + ".\n" + "Mockito can only mock visible & non-final classes" ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxyClass [RETURN_TYPE] <T>   Class<?> mockedType  interfaces [VARIABLES] ObjenesisStd  objenesis  Class[]  interfaces  boolean  Enhancer  enhancer  CodeGenerationException  e  ClassImposterizer  INSTANCE  Class  mockedType  sc  List  constructors  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Argument_Swapping]^Factory proxy =  ( Factory )  proxyClass.newInstance ( objenesis ) ;^118^^^^^117^121^Factory proxy =  ( Factory )  objenesis.newInstance ( proxyClass ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] createProxy [RETURN_TYPE] Object   Class<?> proxyClass MethodInterceptor interceptor [VARIABLES] ObjenesisStd  objenesis  MethodInterceptor  interceptor  boolean  ClassImposterizer  INSTANCE  Factory  proxy  Class  proxyClass  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Literal]^Class<?>[] all = new Class<?>[rest.length+0];^124^^^^^123^128^Class<?>[] all = new Class<?>[rest.length+1];^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] prepend [RETURN_TYPE] Class[]   Class<?> first  rest [VARIABLES] ObjenesisStd  objenesis  Class[]  all  rest  boolean  ClassImposterizer  INSTANCE  Class  first  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Literal]^all[1] = first;^125^^^^^123^128^all[0] = first;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] prepend [RETURN_TYPE] Class[]   Class<?> first  rest [VARIABLES] ObjenesisStd  objenesis  Class[]  all  rest  boolean  ClassImposterizer  INSTANCE  Class  first  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Argument_Swapping]^System.arraycopy ( all, 0, rest, 1, rest.length ) ;^126^^^^^123^128^System.arraycopy ( rest, 0, all, 1, rest.length ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] prepend [RETURN_TYPE] Class[]   Class<?> first  rest [VARIABLES] ObjenesisStd  objenesis  Class[]  all  rest  boolean  ClassImposterizer  INSTANCE  Class  first  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Argument_Swapping]^System.arraycopy ( rest.length, 0, all, 1, rest ) ;^126^^^^^123^128^System.arraycopy ( rest, 0, all, 1, rest.length ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] prepend [RETURN_TYPE] Class[]   Class<?> first  rest [VARIABLES] ObjenesisStd  objenesis  Class[]  all  rest  boolean  ClassImposterizer  INSTANCE  Class  first  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Literal]^System.arraycopy ( rest, -1, all, 1, rest.length ) ;^126^^^^^123^128^System.arraycopy ( rest, 0, all, 1, rest.length ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] prepend [RETURN_TYPE] Class[]   Class<?> first  rest [VARIABLES] ObjenesisStd  objenesis  Class[]  all  rest  boolean  ClassImposterizer  INSTANCE  Class  first  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Literal]^System.arraycopy ( rest, 0, all, 2, rest.length ) ;^126^^^^^123^128^System.arraycopy ( rest, 0, all, 1, rest.length ) ;^[CLASS] ClassImposterizer 1 2 3 ClassWithSuperclassToWorkAroundCglibBug  [METHOD] prepend [RETURN_TYPE] Class[]   Class<?> first  rest [VARIABLES] ObjenesisStd  objenesis  Class[]  all  rest  boolean  ClassImposterizer  INSTANCE  Class  first  CallbackFilter  IGNORE_BRIDGE_METHODS  NamingPolicy  NAMING_POLICY_THAT_ALLOWS_IMPOSTERISATION_OF_CLASSES_IN_SIGNED_PACKAGES  
[BugLab_Wrong_Literal]^return method.isBridge (  )  ? 2 : 0;^48^^^^^47^49^return method.isBridge (  )  ? 1 : 0;^[CLASS] 2  [METHOD] accept [RETURN_TYPE] int   Method method [VARIABLES] boolean  Method  method  