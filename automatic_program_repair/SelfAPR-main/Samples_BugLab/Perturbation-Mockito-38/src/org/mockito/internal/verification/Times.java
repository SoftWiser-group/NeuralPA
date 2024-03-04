[BugLab_Variable_Misuse]^if  ( wantedCount < 0 )  {^25^^^^^24^29^if  ( wantedNumberOfInvocations < 0 )  {^[CLASS] Times  [METHOD] <init> [RETURN_TYPE] Times(int)   int wantedNumberOfInvocations [VARIABLES] int  wantedCount  wantedNumberOfInvocations  boolean  
[BugLab_Wrong_Operator]^if  ( wantedNumberOfInvocations <= 0 )  {^25^^^^^24^29^if  ( wantedNumberOfInvocations < 0 )  {^[CLASS] Times  [METHOD] <init> [RETURN_TYPE] Times(int)   int wantedNumberOfInvocations [VARIABLES] int  wantedCount  wantedNumberOfInvocations  boolean  
[BugLab_Wrong_Literal]^if  ( wantedNumberOfInvocations < wantedCount )  {^25^^^^^24^29^if  ( wantedNumberOfInvocations < 0 )  {^[CLASS] Times  [METHOD] <init> [RETURN_TYPE] Times(int)   int wantedNumberOfInvocations [VARIABLES] int  wantedCount  wantedNumberOfInvocations  boolean  
[BugLab_Variable_Misuse]^if  ( wantedNumberOfInvocations > 0 )  {^32^^^^^31^38^if  ( wantedCount > 0 )  {^[CLASS] Times  [METHOD] verify [RETURN_TYPE] void   VerificationData data [VARIABLES] MissingInvocationChecker  missingInvocation  NumberOfInvocationsChecker  numberOfInvocations  boolean  int  wantedCount  wantedNumberOfInvocations  VerificationData  data  
[BugLab_Wrong_Operator]^if  ( wantedCount >= 0 )  {^32^^^^^31^38^if  ( wantedCount > 0 )  {^[CLASS] Times  [METHOD] verify [RETURN_TYPE] void   VerificationData data [VARIABLES] MissingInvocationChecker  missingInvocation  NumberOfInvocationsChecker  numberOfInvocations  boolean  int  wantedCount  wantedNumberOfInvocations  VerificationData  data  
[BugLab_Variable_Misuse]^numberOfInvocations.check ( data.getAllInvocations (  ) , data.getWanted (  ) , wantedNumberOfInvocations ) ;^37^^^^^31^38^numberOfInvocations.check ( data.getAllInvocations (  ) , data.getWanted (  ) , wantedCount ) ;^[CLASS] Times  [METHOD] verify [RETURN_TYPE] void   VerificationData data [VARIABLES] MissingInvocationChecker  missingInvocation  NumberOfInvocationsChecker  numberOfInvocations  boolean  int  wantedCount  wantedNumberOfInvocations  VerificationData  data  
[BugLab_Argument_Swapping]^numberOfInvocations.check ( wantedCount.getAllInvocations (  ) , data.getWanted (  ) , data ) ;^37^^^^^31^38^numberOfInvocations.check ( data.getAllInvocations (  ) , data.getWanted (  ) , wantedCount ) ;^[CLASS] Times  [METHOD] verify [RETURN_TYPE] void   VerificationData data [VARIABLES] MissingInvocationChecker  missingInvocation  NumberOfInvocationsChecker  numberOfInvocations  boolean  int  wantedCount  wantedNumberOfInvocations  VerificationData  data  
[BugLab_Variable_Misuse]^if  ( wantedNumberOfInvocations > 0 )  {^44^^^^^40^50^if  ( wantedCount > 0 )  {^[CLASS] Times  [METHOD] verifyInOrder [RETURN_TYPE] void   VerificationData data [VARIABLES] boolean  NumberOfInvocationsInOrderChecker  numberOfCalls  InvocationMatcher  wanted  List  allInvocations  int  wantedCount  wantedNumberOfInvocations  VerificationData  data  MissingInvocationInOrderChecker  missingInvocation  
[BugLab_Argument_Swapping]^if  ( wantedCountCount > 0 )  {^44^^^^^40^50^if  ( wantedCount > 0 )  {^[CLASS] Times  [METHOD] verifyInOrder [RETURN_TYPE] void   VerificationData data [VARIABLES] boolean  NumberOfInvocationsInOrderChecker  numberOfCalls  InvocationMatcher  wanted  List  allInvocations  int  wantedCount  wantedNumberOfInvocations  VerificationData  data  MissingInvocationInOrderChecker  missingInvocation  
[BugLab_Wrong_Operator]^if  ( wantedCount >= 0 )  {^44^^^^^40^50^if  ( wantedCount > 0 )  {^[CLASS] Times  [METHOD] verifyInOrder [RETURN_TYPE] void   VerificationData data [VARIABLES] boolean  NumberOfInvocationsInOrderChecker  numberOfCalls  InvocationMatcher  wanted  List  allInvocations  int  wantedCount  wantedNumberOfInvocations  VerificationData  data  MissingInvocationInOrderChecker  missingInvocation  
[BugLab_Argument_Swapping]^missingInvocation.check ( wanted, allInvocations, this ) ;^46^^^^^40^50^missingInvocation.check ( allInvocations, wanted, this ) ;^[CLASS] Times  [METHOD] verifyInOrder [RETURN_TYPE] void   VerificationData data [VARIABLES] boolean  NumberOfInvocationsInOrderChecker  numberOfCalls  InvocationMatcher  wanted  List  allInvocations  int  wantedCount  wantedNumberOfInvocations  VerificationData  data  MissingInvocationInOrderChecker  missingInvocation  
[BugLab_Variable_Misuse]^missingInvocation.check ( this, wanted, this ) ;^46^^^^^40^50^missingInvocation.check ( allInvocations, wanted, this ) ;^[CLASS] Times  [METHOD] verifyInOrder [RETURN_TYPE] void   VerificationData data [VARIABLES] boolean  NumberOfInvocationsInOrderChecker  numberOfCalls  InvocationMatcher  wanted  List  allInvocations  int  wantedCount  wantedNumberOfInvocations  VerificationData  data  MissingInvocationInOrderChecker  missingInvocation  
[BugLab_Variable_Misuse]^numberOfCalls.check ( allInvocations, wanted, wantedNumberOfInvocations ) ;^49^^^^^40^50^numberOfCalls.check ( allInvocations, wanted, wantedCount ) ;^[CLASS] Times  [METHOD] verifyInOrder [RETURN_TYPE] void   VerificationData data [VARIABLES] boolean  NumberOfInvocationsInOrderChecker  numberOfCalls  InvocationMatcher  wanted  List  allInvocations  int  wantedCount  wantedNumberOfInvocations  VerificationData  data  MissingInvocationInOrderChecker  missingInvocation  
[BugLab_Argument_Swapping]^numberOfCalls.check ( allInvocations, wantedCount, wanted ) ;^49^^^^^40^50^numberOfCalls.check ( allInvocations, wanted, wantedCount ) ;^[CLASS] Times  [METHOD] verifyInOrder [RETURN_TYPE] void   VerificationData data [VARIABLES] boolean  NumberOfInvocationsInOrderChecker  numberOfCalls  InvocationMatcher  wanted  List  allInvocations  int  wantedCount  wantedNumberOfInvocations  VerificationData  data  MissingInvocationInOrderChecker  missingInvocation  
[BugLab_Argument_Swapping]^numberOfCalls.check ( wantedCount, wanted, allInvocations ) ;^49^^^^^40^50^numberOfCalls.check ( allInvocations, wanted, wantedCount ) ;^[CLASS] Times  [METHOD] verifyInOrder [RETURN_TYPE] void   VerificationData data [VARIABLES] boolean  NumberOfInvocationsInOrderChecker  numberOfCalls  InvocationMatcher  wanted  List  allInvocations  int  wantedCount  wantedNumberOfInvocations  VerificationData  data  MissingInvocationInOrderChecker  missingInvocation  
[BugLab_Variable_Misuse]^return "Wanted invocations count: " + wantedNumberOfInvocations;^54^^^^^53^55^return "Wanted invocations count: " + wantedCount;^[CLASS] Times  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] int  wantedCount  wantedNumberOfInvocations  boolean  
