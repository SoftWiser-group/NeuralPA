[BugLab_Variable_Misuse]^this.unstubbedInvocations = new LinkedList<InvocationMatcher> ( null ) ;^24^^^^^22^25^this.unstubbedInvocations = new LinkedList<InvocationMatcher> ( unstubbedInvocations ) ;^[CLASS] WarningsPrinter  [METHOD] <init> [RETURN_TYPE] List)   Invocation> unusedStubs InvocationMatcher> unstubbedInvocations [VARIABLES] List  unstubbedInvocations  unusedStubs  boolean  
[BugLab_Variable_Misuse]^while ( iIterator.hasNext (  )  )  {^29^^^^^27^49^while ( unusedIterator.hasNext (  )  )  {^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Variable_Misuse]^while ( i1Iterator.hasNext (  )  )  {^32^^^^^27^49^while ( unstubbedIterator.hasNext (  )  )  {^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Argument_Swapping]^while ( unstubbed.hasNext (  )  )  {^32^^^^^27^49^while ( unstubbedIterator.hasNext (  )  )  {^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Variable_Misuse]^if ( i1.hasSimilarMethod ( unused )  )  {^34^^^^^27^49^if ( unstubbed.hasSimilarMethod ( unused )  )  {^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Variable_Misuse]^logger.log ( stubbedMethodCalledWithDifferentArguments ( unused, i1 )  ) ;^35^^^^^27^49^logger.log ( stubbedMethodCalledWithDifferentArguments ( unused, unstubbed )  ) ;^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Argument_Swapping]^logger.log ( stubbedMethodCalledWithDifferentArguments ( unstubbed, unused )  ) ;^35^^^^^27^49^logger.log ( stubbedMethodCalledWithDifferentArguments ( unused, unstubbed )  ) ;^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Variable_Misuse]^logger.log ( stubbedMethodCalledWithDifferentArguments ( i, unstubbed )  ) ;^35^^^^^27^49^logger.log ( stubbedMethodCalledWithDifferentArguments ( unused, unstubbed )  ) ;^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Variable_Misuse]^if ( unstubbed.hasSimilarMethod ( i )  )  {^34^^^^^27^49^if ( unstubbed.hasSimilarMethod ( unused )  )  {^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Argument_Swapping]^if ( unused.hasSimilarMethod ( unstubbed )  )  {^34^^^^^27^49^if ( unstubbed.hasSimilarMethod ( unused )  )  {^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Variable_Misuse]^while ( null.hasNext (  )  )  {^32^^^^^27^49^while ( unstubbedIterator.hasNext (  )  )  {^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Argument_Swapping]^while ( unstubbedIteratorIterator.hasNext (  )  )  {^32^^^^^27^49^while ( unstubbedIterator.hasNext (  )  )  {^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Variable_Misuse]^Iterator<InvocationMatcher> unstubbedIterator = null.iterator (  ) ;^31^^^^^27^49^Iterator<InvocationMatcher> unstubbedIterator = unstubbedInvocations.iterator (  ) ;^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Variable_Misuse]^logger.log ( thisStubWasNotUsed ( unused )  ) ;^43^^^^^27^49^logger.log ( thisStubWasNotUsed ( i )  ) ;^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[BugLab_Variable_Misuse]^logger.log ( thisMethodWasNotStubbed ( unstubbed )  ) ;^47^^^^^27^49^logger.log ( thisMethodWasNotStubbed ( i1 )  ) ;^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
