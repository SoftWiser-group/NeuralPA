[BugLab_Variable_Misuse]^this ( oldMockHandler.mockName, mockingProgress, oldMockHandler.matchersBinder, oldMockHandler.mockSettings ) ;^54^^^^^53^55^this ( oldMockHandler.mockName, oldMockHandler.mockingProgress, oldMockHandler.matchersBinder, oldMockHandler.mockSettings ) ;^[CLASS] MockHandler  [METHOD] <init> [RETURN_TYPE] MockHandler)   MockHandler<T> oldMockHandler [VARIABLES] RegisteredInvocations  registeredInvocations  MockName  mockName  boolean  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  MockHandler  oldMockHandler  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  
[BugLab_Variable_Misuse]^this ( oldMockHandler.mockName, oldMockHandler.mockingProgress, oldMockHandler.matchersBinder, mockSettings ) ;^54^^^^^53^55^this ( oldMockHandler.mockName, oldMockHandler.mockingProgress, oldMockHandler.matchersBinder, oldMockHandler.mockSettings ) ;^[CLASS] MockHandler  [METHOD] <init> [RETURN_TYPE] MockHandler)   MockHandler<T> oldMockHandler [VARIABLES] RegisteredInvocations  registeredInvocations  MockName  mockName  boolean  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  MockHandler  oldMockHandler  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  
[BugLab_Argument_Swapping]^this ( oldMockHandler.mockingProgress.mockName, oldMockHandler, oldMockHandler.matchersBinder, oldMockHandler.mockSettings ) ;^54^^^^^53^55^this ( oldMockHandler.mockName, oldMockHandler.mockingProgress, oldMockHandler.matchersBinder, oldMockHandler.mockSettings ) ;^[CLASS] MockHandler  [METHOD] <init> [RETURN_TYPE] MockHandler)   MockHandler<T> oldMockHandler [VARIABLES] RegisteredInvocations  registeredInvocations  MockName  mockName  boolean  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  MockHandler  oldMockHandler  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  
[BugLab_Argument_Swapping]^this ( oldMockHandler, oldMockHandler.mockName.mockingProgress, oldMockHandler.matchersBinder, oldMockHandler.mockSettings ) ;^54^^^^^53^55^this ( oldMockHandler.mockName, oldMockHandler.mockingProgress, oldMockHandler.matchersBinder, oldMockHandler.mockSettings ) ;^[CLASS] MockHandler  [METHOD] <init> [RETURN_TYPE] MockHandler)   MockHandler<T> oldMockHandler [VARIABLES] RegisteredInvocations  registeredInvocations  MockName  mockName  boolean  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  MockHandler  oldMockHandler  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  
[BugLab_Argument_Swapping]^this ( oldMockHandler.mockName, oldMockHandler.mockSettings, oldMockHandler.matchersBinder, oldMockHandler.mockingProgress ) ;^54^^^^^53^55^this ( oldMockHandler.mockName, oldMockHandler.mockingProgress, oldMockHandler.matchersBinder, oldMockHandler.mockSettings ) ;^[CLASS] MockHandler  [METHOD] <init> [RETURN_TYPE] MockHandler)   MockHandler<T> oldMockHandler [VARIABLES] RegisteredInvocations  registeredInvocations  MockName  mockName  boolean  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  MockHandler  oldMockHandler  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  
[BugLab_Argument_Swapping]^this ( oldMockHandler.matchersBinder.mockName, oldMockHandler.mockingProgress, oldMockHandler, oldMockHandler.mockSettings ) ;^54^^^^^53^55^this ( oldMockHandler.mockName, oldMockHandler.mockingProgress, oldMockHandler.matchersBinder, oldMockHandler.mockSettings ) ;^[CLASS] MockHandler  [METHOD] <init> [RETURN_TYPE] MockHandler)   MockHandler<T> oldMockHandler [VARIABLES] RegisteredInvocations  registeredInvocations  MockName  mockName  boolean  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  MockHandler  oldMockHandler  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  
[BugLab_Variable_Misuse]^Invocation invocation = new Invocation ( ret, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^60^^^^^45^75^Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Invocation invocation = new Invocation ( method, proxy, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^60^^^^^45^75^Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Invocation invocation = new Invocation ( proxy, method, methodProxy, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( args )  ) ;^60^^^^^45^75^Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Invocation invocation = new Invocation ( methodProxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( proxy )  ) ;^60^^^^^45^75^Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Invocation invocation = new Invocation ( args, method, proxy, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^60^^^^^45^75^Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Invocation invocation = new Invocation ( proxy, methodProxy, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( method )  ) ;^60^^^^^45^75^Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^InvocationMatcher mockingProgressMatcher = matchersBinder.bindMatchers ( invocation.getArgumentMatcherStorage (  ) , invocation ) ;^61^^^^^46^76^InvocationMatcher invocationMatcher = matchersBinder.bindMatchers ( mockingProgress.getArgumentMatcherStorage (  ) , invocation ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^InvocationMatcher matchersBinderMatcher = invocation.bindMatchers ( mockingProgress.getArgumentMatcherStorage (  ) , invocation ) ;^61^^^^^46^76^InvocationMatcher invocationMatcher = matchersBinder.bindMatchers ( mockingProgress.getArgumentMatcherStorage (  ) , invocation ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^InvocationMatcher invocationMatcher = mockingProgress.bindMatchers ( matchersBinder.getArgumentMatcherStorage (  ) , invocation ) ;^61^^^^^46^76^InvocationMatcher invocationMatcher = matchersBinder.bindMatchers ( mockingProgress.getArgumentMatcherStorage (  ) , invocation ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Invocation invocation = new Invocation ( proxy, args, method, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^60^^^^^45^75^Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Variable_Misuse]^Invocation invocation = new Invocation ( ret, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^67^^^^^52^82^Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Invocation invocation = new Invocation ( method, proxy, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^67^^^^^52^82^Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Invocation invocation = new Invocation ( args, method, proxy, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^67^^^^^52^82^Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Invocation invocation = new Invocation ( proxy, method, methodProxy, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( args )  ) ;^67^^^^^52^82^Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Invocation invocation = new Invocation ( proxy, methodProxy, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( method )  ) ;^67^^^^^52^82^Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^InvocationMatcher mockingProgressMatcher = matchersBinder.bindMatchers ( invocation.getArgumentMatcherStorage (  ) , invocation ) ;^68^^^^^53^83^InvocationMatcher invocationMatcher = matchersBinder.bindMatchers ( mockingProgress.getArgumentMatcherStorage (  ) , invocation ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^InvocationMatcher matchersBinderMatcher = invocation.bindMatchers ( mockingProgress.getArgumentMatcherStorage (  ) , invocation ) ;^68^^^^^53^83^InvocationMatcher invocationMatcher = matchersBinder.bindMatchers ( mockingProgress.getArgumentMatcherStorage (  ) , invocation ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^InvocationMatcher invocationMatcher = mockingProgress.bindMatchers ( matchersBinder.getArgumentMatcherStorage (  ) , invocation ) ;^68^^^^^53^83^InvocationMatcher invocationMatcher = matchersBinder.bindMatchers ( mockingProgress.getArgumentMatcherStorage (  ) , invocation ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Wrong_Operator]^if  ( verificationMode == null )  {^72^^^^^57^87^if  ( verificationMode != null )  {^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^VerificationDataImpl data = new VerificationDataImpl ( invocationMatcher.getAll (  ) , registeredInvocations ) ;^73^^^^^58^88^VerificationDataImpl data = new VerificationDataImpl ( registeredInvocations.getAll (  ) , invocationMatcher ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^OngoingStubbingImpl<T> ongoingStubbing = new OngoingStubbingImpl<T> ( registeredInvocations, mockitoStubber ) ;^80^^^^^65^95^OngoingStubbingImpl<T> ongoingStubbing = new OngoingStubbingImpl<T> ( mockitoStubber, registeredInvocations ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Answer<?> stubbedAnswer = invocation.findAnswerFor ( mockitoStubber ) ;^83^^^^^68^98^Answer<?> stubbedAnswer = mockitoStubber.findAnswerFor ( invocation ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Wrong_Operator]^if  ( !invocation.isVoid (  )  || stubbedAnswer == null )  {^84^^^^^69^99^if  ( !invocation.isVoid (  )  && stubbedAnswer == null )  {^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Wrong_Operator]^if  ( !invocation.isVoid (  )  && stubbedAnswer != null )  {^84^^^^^69^99^if  ( !invocation.isVoid (  )  && stubbedAnswer == null )  {^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Wrong_Operator]^if  ( stubbedAnswer == null )  {^89^^^^^74^104^if  ( stubbedAnswer != null )  {^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Variable_Misuse]^return proxy;^99^^^^^89^100^return ret;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Object ret = invocation.getDefaultAnswer (  ) .answer ( mockSettings ) ;^93^^^^^89^100^Object ret = mockSettings.getDefaultAnswer (  ) .answer ( invocation ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^return invocation.answer ( stubbedAnswer ) ;^91^^^^^76^106^return stubbedAnswer.answer ( invocation ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Variable_Misuse]^return proxy;^99^^^^^84^114^return ret;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^Object ret = invocation.getDefaultAnswer (  ) .answer ( mockSettings ) ;^93^^^^^78^108^Object ret = mockSettings.getDefaultAnswer (  ) .answer ( invocation ) ;^[CLASS] MockHandler  [METHOD] intercept [RETURN_TYPE] Object   Object proxy Method method Object[] args MethodProxy methodProxy [VARIABLES] VerificationDataImpl  data  RegisteredInvocations  registeredInvocations  Invocation  invocation  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  InvocationMatcher  invocationMatcher  Method  method  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  VerificationMode  verificationMode  Answer  stubbedAnswer  MockName  mockName  boolean  OngoingStubbingImpl  ongoingStubbing  MethodProxy  methodProxy  Object  proxy  ret  Object[]  args  
[BugLab_Argument_Swapping]^return new VoidMethodStubbableImpl<T> ( mockitoStubber, mock ) ;^109^^^^^108^110^return new VoidMethodStubbableImpl<T> ( mock, mockitoStubber ) ;^[CLASS] MockHandler  [METHOD] voidMethodStubbable [RETURN_TYPE] VoidMethodStubbable   T mock [VARIABLES] RegisteredInvocations  registeredInvocations  MockName  mockName  boolean  T  mock  MatchersBinder  matchersBinder  MockSettingsImpl  mockSettings  MockingProgress  mockingProgress  MockitoStubber  mockitoStubber  
