[P14_Delete_Statement]^^17^18^^^^16^22^List<Matcher> lastMatchers = argumentMatcherStorage.pullMatchers (  ) ; validateMatchers ( invocation, lastMatchers ) ;^[CLASS] MatchersBinder  [METHOD] bindMatchers [RETURN_TYPE] InvocationMatcher   ArgumentMatcherStorage argumentMatcherStorage Invocation invocation [VARIABLES] boolean  Invocation  invocation  InvocationMatcher  invocationWithMatchers  List  lastMatchers  ArgumentMatcherStorage  argumentMatcherStorage  
[P5_Replace_Variable]^validateMatchers (  lastMatchers ) ;^18^^^^^16^22^validateMatchers ( invocation, lastMatchers ) ;^[CLASS] MatchersBinder  [METHOD] bindMatchers [RETURN_TYPE] InvocationMatcher   ArgumentMatcherStorage argumentMatcherStorage Invocation invocation [VARIABLES] boolean  Invocation  invocation  InvocationMatcher  invocationWithMatchers  List  lastMatchers  ArgumentMatcherStorage  argumentMatcherStorage  
[P5_Replace_Variable]^validateMatchers ( invocation ) ;^18^^^^^16^22^validateMatchers ( invocation, lastMatchers ) ;^[CLASS] MatchersBinder  [METHOD] bindMatchers [RETURN_TYPE] InvocationMatcher   ArgumentMatcherStorage argumentMatcherStorage Invocation invocation [VARIABLES] boolean  Invocation  invocation  InvocationMatcher  invocationWithMatchers  List  lastMatchers  ArgumentMatcherStorage  argumentMatcherStorage  
[P5_Replace_Variable]^validateMatchers ( lastMatchers, invocation ) ;^18^^^^^16^22^validateMatchers ( invocation, lastMatchers ) ;^[CLASS] MatchersBinder  [METHOD] bindMatchers [RETURN_TYPE] InvocationMatcher   ArgumentMatcherStorage argumentMatcherStorage Invocation invocation [VARIABLES] boolean  Invocation  invocation  InvocationMatcher  invocationWithMatchers  List  lastMatchers  ArgumentMatcherStorage  argumentMatcherStorage  
[P7_Replace_Invocation]^bindMatchers ( invocation, lastMatchers ) ;^18^^^^^16^22^validateMatchers ( invocation, lastMatchers ) ;^[CLASS] MatchersBinder  [METHOD] bindMatchers [RETURN_TYPE] InvocationMatcher   ArgumentMatcherStorage argumentMatcherStorage Invocation invocation [VARIABLES] boolean  Invocation  invocation  InvocationMatcher  invocationWithMatchers  List  lastMatchers  ArgumentMatcherStorage  argumentMatcherStorage  
[P14_Delete_Statement]^^18^^^^^16^22^validateMatchers ( invocation, lastMatchers ) ;^[CLASS] MatchersBinder  [METHOD] bindMatchers [RETURN_TYPE] InvocationMatcher   ArgumentMatcherStorage argumentMatcherStorage Invocation invocation [VARIABLES] boolean  Invocation  invocation  InvocationMatcher  invocationWithMatchers  List  lastMatchers  ArgumentMatcherStorage  argumentMatcherStorage  
[P4_Replace_Constructor]^InvocationMatcher invocationWithMatchers = new InvocationMatcher (  lastMatchers ) ;^20^^^^^16^22^InvocationMatcher invocationWithMatchers = new InvocationMatcher ( invocation, lastMatchers ) ;^[CLASS] MatchersBinder  [METHOD] bindMatchers [RETURN_TYPE] InvocationMatcher   ArgumentMatcherStorage argumentMatcherStorage Invocation invocation [VARIABLES] boolean  Invocation  invocation  InvocationMatcher  invocationWithMatchers  List  lastMatchers  ArgumentMatcherStorage  argumentMatcherStorage  
[P4_Replace_Constructor]^InvocationMatcher invocationWithMatchers = new InvocationMatcher ( invocation ) ;^20^^^^^16^22^InvocationMatcher invocationWithMatchers = new InvocationMatcher ( invocation, lastMatchers ) ;^[CLASS] MatchersBinder  [METHOD] bindMatchers [RETURN_TYPE] InvocationMatcher   ArgumentMatcherStorage argumentMatcherStorage Invocation invocation [VARIABLES] boolean  Invocation  invocation  InvocationMatcher  invocationWithMatchers  List  lastMatchers  ArgumentMatcherStorage  argumentMatcherStorage  
[P5_Replace_Variable]^InvocationMatcher lastMatchersWithMatchers = new InvocationMatcher ( invocation, invocation ) ;^20^^^^^16^22^InvocationMatcher invocationWithMatchers = new InvocationMatcher ( invocation, lastMatchers ) ;^[CLASS] MatchersBinder  [METHOD] bindMatchers [RETURN_TYPE] InvocationMatcher   ArgumentMatcherStorage argumentMatcherStorage Invocation invocation [VARIABLES] boolean  Invocation  invocation  InvocationMatcher  invocationWithMatchers  List  lastMatchers  ArgumentMatcherStorage  argumentMatcherStorage  
[P15_Unwrap_Block]^int recordedMatchersSize = matchers.size(); int expectedMatchersSize = invocation.getArgumentsCount(); if (expectedMatchersSize != recordedMatchersSize) {    new org.mockito.exceptions.Reporter().invalidUseOfMatchers(expectedMatchersSize, recordedMatchersSize);};^25^26^27^28^29^24^32^if  ( !matchers.isEmpty (  )  )  { int recordedMatchersSize = matchers.size (  ) ; int expectedMatchersSize = invocation.getArgumentsCount (  ) ; if  ( expectedMatchersSize != recordedMatchersSize )  { new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ; }^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P16_Remove_Block]^^25^26^27^28^29^24^32^if  ( !matchers.isEmpty (  )  )  { int recordedMatchersSize = matchers.size (  ) ; int expectedMatchersSize = invocation.getArgumentsCount (  ) ; if  ( expectedMatchersSize != recordedMatchersSize )  { new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ; }^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P13_Insert_Block]^if  ( expectedMatchersSize != recordedMatchersSize )  {     new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ; }^25^^^^^24^32^[Delete]^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P2_Replace_Operator]^if  ( expectedMatchersSize == recordedMatchersSize )  {^28^^^^^24^32^if  ( expectedMatchersSize != recordedMatchersSize )  {^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P5_Replace_Variable]^if  ( recordedMatchersSize != expectedMatchersSize )  {^28^^^^^24^32^if  ( expectedMatchersSize != recordedMatchersSize )  {^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P8_Replace_Mix]^if  ( expectedMatchersSize = recordedMatchersSize )  {^28^^^^^24^32^if  ( expectedMatchersSize != recordedMatchersSize )  {^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P15_Unwrap_Block]^new org.mockito.exceptions.Reporter().invalidUseOfMatchers(expectedMatchersSize, recordedMatchersSize);^28^29^30^^^24^32^if  ( expectedMatchersSize != recordedMatchersSize )  { new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ; }^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P16_Remove_Block]^^28^29^30^^^24^32^if  ( expectedMatchersSize != recordedMatchersSize )  { new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ; }^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P13_Insert_Block]^if  ( ! ( matchers.isEmpty (  )  )  )  {     int recordedMatchersSize = matchers.size (  ) ;     int expectedMatchersSize = invocation.getArgumentsCount (  ) ;     if  ( expectedMatchersSize != recordedMatchersSize )  {         new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ;     } }^28^^^^^24^32^[Delete]^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P12_Insert_Condition]^if  ( expectedMatchersSize != recordedMatchersSize )  { new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ; }^29^^^^^24^32^new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P4_Replace_Constructor]^new Reporter (  ) .invalidUseOfMatchers (  recordedMatchersSize ) ;^29^^^^^24^32^new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P4_Replace_Constructor]^new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize ) ;^29^^^^^24^32^new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P5_Replace_Variable]^new Reporter (  ) .invalidUseOfMatchers ( recordedMatchersSize, expectedMatchersSize ) ;^29^^^^^24^32^new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P14_Delete_Statement]^^29^^^^^24^32^new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P13_Insert_Block]^if  ( expectedMatchersSize != recordedMatchersSize )  {     new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ; }^29^^^^^24^32^[Delete]^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P1_Replace_Type]^long  recordedMatchersSize = matchers.size (  ) ;^26^^^^^24^32^int recordedMatchersSize = matchers.size (  ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P3_Replace_Literal]^int recordedMatchersSize = matchers.size() - 5 ;^26^^^^^24^32^int recordedMatchersSize = matchers.size (  ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P8_Replace_Mix]^int recordedMatchersSize = matchers .isEmpty (  )  ;^26^^^^^24^32^int recordedMatchersSize = matchers.size (  ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P11_Insert_Donor_Statement]^int expectedMatchersSize = invocation.getArgumentsCount (  ) ;int recordedMatchersSize = matchers.size (  ) ;^26^^^^^24^32^int recordedMatchersSize = matchers.size (  ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P12_Insert_Condition]^if  ( expectedMatchersSize != recordedMatchersSize )  { int recordedMatchersSize = matchers.size (  ) ; }^26^^^^^24^32^int recordedMatchersSize = matchers.size (  ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P1_Replace_Type]^short  expectedMatchersSize = invocation.getArgumentsCount (  ) ;^27^^^^^24^32^int expectedMatchersSize = invocation.getArgumentsCount (  ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P11_Insert_Donor_Statement]^int recordedMatchersSize = matchers.size (  ) ;int expectedMatchersSize = invocation.getArgumentsCount (  ) ;^27^^^^^24^32^int expectedMatchersSize = invocation.getArgumentsCount (  ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P3_Replace_Literal]^int recordedMatchersSize = matchers.size() - 8 ;^26^^^^^24^32^int recordedMatchersSize = matchers.size (  ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P14_Delete_Statement]^^26^27^^^^24^32^int recordedMatchersSize = matchers.size (  ) ; int expectedMatchersSize = invocation.getArgumentsCount (  ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P14_Delete_Statement]^^27^^^^^24^32^int expectedMatchersSize = invocation.getArgumentsCount (  ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P7_Replace_Invocation]^new Reporter (  )  .Reporter (  )  ;^29^^^^^24^32^new Reporter (  ) .invalidUseOfMatchers ( expectedMatchersSize, recordedMatchersSize ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
[P3_Replace_Literal]^int recordedMatchersSize = matchers.size() + 2 ;^26^^^^^24^32^int recordedMatchersSize = matchers.size (  ) ;^[CLASS] MatchersBinder  [METHOD] validateMatchers [RETURN_TYPE] void   Invocation invocation Matcher> matchers [VARIABLES] boolean  Invocation  invocation  List  matchers  int  expectedMatchersSize  recordedMatchersSize  
