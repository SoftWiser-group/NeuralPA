[BugLab_Variable_Misuse]^return  ( isRunner || fromOrgMockito )  && !isRunner && !isInternalRunner;^19^^^^^14^20^return  ( fromMockObject || fromOrgMockito )  && !isRunner && !isInternalRunner;^[CLASS] StackTraceFilter  [METHOD] isBad [RETURN_TYPE] boolean   StackTraceElement e [VARIABLES] boolean  fromMockObject  fromOrgMockito  isInternalRunner  isRunner  StackTraceElement  e  
[BugLab_Variable_Misuse]^return  ( fromMockObject || isInternalRunner )  && !isRunner && !isInternalRunner;^19^^^^^14^20^return  ( fromMockObject || fromOrgMockito )  && !isRunner && !isInternalRunner;^[CLASS] StackTraceFilter  [METHOD] isBad [RETURN_TYPE] boolean   StackTraceElement e [VARIABLES] boolean  fromMockObject  fromOrgMockito  isInternalRunner  isRunner  StackTraceElement  e  
[BugLab_Variable_Misuse]^return  ( fromMockObject || fromOrgMockito )  && !fromOrgMockito && !isInternalRunner;^19^^^^^14^20^return  ( fromMockObject || fromOrgMockito )  && !isRunner && !isInternalRunner;^[CLASS] StackTraceFilter  [METHOD] isBad [RETURN_TYPE] boolean   StackTraceElement e [VARIABLES] boolean  fromMockObject  fromOrgMockito  isInternalRunner  isRunner  StackTraceElement  e  
[BugLab_Argument_Swapping]^return  ( fromOrgMockito || fromMockObject )  && !isRunner && !isInternalRunner;^19^^^^^14^20^return  ( fromMockObject || fromOrgMockito )  && !isRunner && !isInternalRunner;^[CLASS] StackTraceFilter  [METHOD] isBad [RETURN_TYPE] boolean   StackTraceElement e [VARIABLES] boolean  fromMockObject  fromOrgMockito  isInternalRunner  isRunner  StackTraceElement  e  
[BugLab_Wrong_Operator]^return  ( fromMockObject || fromOrgMockito )  || !isRunner && !isInternalRunner;^19^^^^^14^20^return  ( fromMockObject || fromOrgMockito )  && !isRunner && !isInternalRunner;^[CLASS] StackTraceFilter  [METHOD] isBad [RETURN_TYPE] boolean   StackTraceElement e [VARIABLES] boolean  fromMockObject  fromOrgMockito  isInternalRunner  isRunner  StackTraceElement  e  
[BugLab_Wrong_Operator]^return  ( fromMockObject && fromOrgMockito )  && !isRunner && !isInternalRunner;^19^^^^^14^20^return  ( fromMockObject || fromOrgMockito )  && !isRunner && !isInternalRunner;^[CLASS] StackTraceFilter  [METHOD] isBad [RETURN_TYPE] boolean   StackTraceElement e [VARIABLES] boolean  fromMockObject  fromOrgMockito  isInternalRunner  isRunner  StackTraceElement  e  
[BugLab_Wrong_Literal]^int firstBad = -i;^32^^^^^27^54^int firstBad = -1;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Variable_Misuse]^if  ( !this.isBad ( unfilteredStackTrace.get ( lastBad )  )  )  {^34^^^^^27^54^if  ( !this.isBad ( unfilteredStackTrace.get ( i )  )  )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Argument_Swapping]^if  ( !this.isBad ( i.get ( unfilteredStackTrace )  )  )  {^34^^^^^27^54^if  ( !this.isBad ( unfilteredStackTrace.get ( i )  )  )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Variable_Misuse]^if  ( lastBad == -1 )  {^38^^^^^27^54^if  ( firstBad == -1 )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Operator]^if  ( firstBad != -1 )  {^38^^^^^27^54^if  ( firstBad == -1 )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Literal]^if  ( firstBad == -2 )  {^38^^^^^27^54^if  ( firstBad == -1 )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Variable_Misuse]^firstBad = lastBad;^39^^^^^27^54^firstBad = i;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Variable_Misuse]^for  ( lastBadnt i = 0; i < unfilteredStackTrace.size (  ) ; i++ )  {^33^^^^^27^54^for  ( int i = 0; i < unfilteredStackTrace.size (  ) ; i++ )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Argument_Swapping]^for  ( unfilteredStackTracent i = 0; i < i.size (  ) ; i++ )  {^33^^^^^27^54^for  ( int i = 0; i < unfilteredStackTrace.size (  ) ; i++ )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= unfilteredStackTrace.size (  ) ; i++ )  {^33^^^^^27^54^for  ( int i = 0; i < unfilteredStackTrace.size (  ) ; i++ )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Literal]^for  ( int i = 1; i < unfilteredStackTrace.size (  ) ; i++ )  {^33^^^^^27^54^for  ( int i = 0; i < unfilteredStackTrace.size (  ) ; i++ )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Operator]^if  ( firstBad >= -1 )  {^38^^^^^27^54^if  ( firstBad == -1 )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Literal]^if  ( firstBad == - )  {^38^^^^^27^54^if  ( firstBad == -1 )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Variable_Misuse]^lastBad = lastBad;^37^^^^^27^54^lastBad = i;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Variable_Misuse]^if  ( keepTop && lastBad != -1 )  {^44^^^^^27^54^if  ( keepTop && firstBad != -1 )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Argument_Swapping]^if  ( firstBad && keepTop != -1 )  {^44^^^^^27^54^if  ( keepTop && firstBad != -1 )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Operator]^if  ( keepTop || firstBad != -1 )  {^44^^^^^27^54^if  ( keepTop && firstBad != -1 )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Operator]^if  ( keepTop && firstBad <= -1 )  {^44^^^^^27^54^if  ( keepTop && firstBad != -1 )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Literal]^if  ( keepTop && firstBad != -firstBad )  {^44^^^^^27^54^if  ( keepTop && firstBad != -1 )  {^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Variable_Misuse]^top = unfilteredStackTrace.subList ( 0, lastBad ) ;^45^^^^^27^54^top = unfilteredStackTrace.subList ( 0, firstBad ) ;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Argument_Swapping]^top = firstBad.subList ( 0, unfilteredStackTrace ) ;^45^^^^^27^54^top = unfilteredStackTrace.subList ( 0, firstBad ) ;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Literal]^top = unfilteredStackTrace.subList ( lastBad, firstBad ) ;^45^^^^^27^54^top = unfilteredStackTrace.subList ( 0, firstBad ) ;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Literal]^top = unfilteredStackTrace.subList ( i, firstBad ) ;^45^^^^^27^54^top = unfilteredStackTrace.subList ( 0, firstBad ) ;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Variable_Misuse]^List<StackTraceElement> bottom = unfilteredStackTrace.subList ( i + 1, unfilteredStackTrace.size (  )  ) ;^50^^^^^27^54^List<StackTraceElement> bottom = unfilteredStackTrace.subList ( lastBad + 1, unfilteredStackTrace.size (  )  ) ;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Argument_Swapping]^List<StackTraceElement> bottom = lastBad.subList ( unfilteredStackTrace + 1, unfilteredStackTrace.size (  )  ) ;^50^^^^^27^54^List<StackTraceElement> bottom = unfilteredStackTrace.subList ( lastBad + 1, unfilteredStackTrace.size (  )  ) ;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Operator]^List<StackTraceElement> bottom = unfilteredStackTrace.subList ( lastBad + 1, unfilteredStackTrace.size (  &  )  ) ;^50^^^^^27^54^List<StackTraceElement> bottom = unfilteredStackTrace.subList ( lastBad + 1, unfilteredStackTrace.size (  )  ) ;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Literal]^List<StackTraceElement> bottom = unfilteredStackTrace.subList ( lastBad + firstBad, unfilteredStackTrace.size (  )  ) ;^50^^^^^27^54^List<StackTraceElement> bottom = unfilteredStackTrace.subList ( lastBad + 1, unfilteredStackTrace.size (  )  ) ;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Operator]^List<StackTraceElement> bottom = unfilteredStackTrace.subList ( lastBad + 1, unfilteredStackTrace.size (  <<  )  ) ;^50^^^^^27^54^List<StackTraceElement> bottom = unfilteredStackTrace.subList ( lastBad + 1, unfilteredStackTrace.size (  )  ) ;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  
[BugLab_Wrong_Literal]^List<StackTraceElement> bottom = unfilteredStackTrace.subList ( lastBad + i, unfilteredStackTrace.size (  )  ) ;^50^^^^^27^54^List<StackTraceElement> bottom = unfilteredStackTrace.subList ( lastBad + 1, unfilteredStackTrace.size (  )  ) ;^[CLASS] StackTraceFilter  [METHOD] filter [RETURN_TYPE] StackTraceElement[]   StackTraceElement[] target boolean keepTop [VARIABLES] boolean  keepTop  StackTraceElement[]  target  List  bottom  filtered  top  unfilteredStackTrace  int  firstBad  i  lastBad  