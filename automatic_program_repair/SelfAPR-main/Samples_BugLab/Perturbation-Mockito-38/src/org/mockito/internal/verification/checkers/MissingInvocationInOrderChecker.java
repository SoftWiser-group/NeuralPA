[BugLab_Argument_Swapping]^List<Invocation> chunk = finder.findAllMatchingUnverifiedChunks ( wanted, invocations ) ;^30^^^^^29^42^List<Invocation> chunk = finder.findAllMatchingUnverifiedChunks ( invocations, wanted ) ;^[CLASS] MissingInvocationInOrderChecker  [METHOD] check [RETURN_TYPE] void   Invocation> invocations InvocationMatcher wanted VerificationMode mode [VARIABLES] boolean  Invocation  previousInOrder  InvocationsFinder  finder  Reporter  reporter  InvocationMatcher  wanted  List  chunk  invocations  VerificationMode  mode  
[BugLab_Argument_Swapping]^List<Invocation> chunk = wanted.findAllMatchingUnverifiedChunks ( invocations, finder ) ;^30^^^^^29^42^List<Invocation> chunk = finder.findAllMatchingUnverifiedChunks ( invocations, wanted ) ;^[CLASS] MissingInvocationInOrderChecker  [METHOD] check [RETURN_TYPE] void   Invocation> invocations InvocationMatcher wanted VerificationMode mode [VARIABLES] boolean  Invocation  previousInOrder  InvocationsFinder  finder  Reporter  reporter  InvocationMatcher  wanted  List  chunk  invocations  VerificationMode  mode  
[BugLab_Argument_Swapping]^Invocation previousInOrder = invocations.findPreviousVerifiedInOrder ( finder ) ;^36^^^^^29^42^Invocation previousInOrder = finder.findPreviousVerifiedInOrder ( invocations ) ;^[CLASS] MissingInvocationInOrderChecker  [METHOD] check [RETURN_TYPE] void   Invocation> invocations InvocationMatcher wanted VerificationMode mode [VARIABLES] boolean  Invocation  previousInOrder  InvocationsFinder  finder  Reporter  reporter  InvocationMatcher  wanted  List  chunk  invocations  VerificationMode  mode  
[BugLab_Variable_Misuse]^Invocation previousInOrder = finder.findPreviousVerifiedInOrder ( 2 ) ;^36^^^^^29^42^Invocation previousInOrder = finder.findPreviousVerifiedInOrder ( invocations ) ;^[CLASS] MissingInvocationInOrderChecker  [METHOD] check [RETURN_TYPE] void   Invocation> invocations InvocationMatcher wanted VerificationMode mode [VARIABLES] boolean  Invocation  previousInOrder  InvocationsFinder  finder  Reporter  reporter  InvocationMatcher  wanted  List  chunk  invocations  VerificationMode  mode  
[BugLab_Wrong_Operator]^if  ( previousInOrder != null )  {^37^^^^^29^42^if  ( previousInOrder == null )  {^[CLASS] MissingInvocationInOrderChecker  [METHOD] check [RETURN_TYPE] void   Invocation> invocations InvocationMatcher wanted VerificationMode mode [VARIABLES] boolean  Invocation  previousInOrder  InvocationsFinder  finder  Reporter  reporter  InvocationMatcher  wanted  List  chunk  invocations  VerificationMode  mode  
[BugLab_Argument_Swapping]^reporter.wantedButNotInvokedInOrder ( previousInOrder, wanted ) ;^40^^^^^29^42^reporter.wantedButNotInvokedInOrder ( wanted, previousInOrder ) ;^[CLASS] MissingInvocationInOrderChecker  [METHOD] check [RETURN_TYPE] void   Invocation> invocations InvocationMatcher wanted VerificationMode mode [VARIABLES] boolean  Invocation  previousInOrder  InvocationsFinder  finder  Reporter  reporter  InvocationMatcher  wanted  List  chunk  invocations  VerificationMode  mode  