[BugLab_Wrong_Operator]^if  ( super.getMessage (  )  == null )  {^101^^^^^100^108^if  ( super.getMessage (  )  != null )  {^[CLASS] NestableError  [METHOD] getMessage [RETURN_TYPE] String   [VARIABLES] Throwable  cause  NestableDelegate  delegate  boolean  
[BugLab_Wrong_Operator]^} else if  ( cause == null )  {^103^^^^^100^108^} else if  ( cause != null )  {^[CLASS] NestableError  [METHOD] getMessage [RETURN_TYPE] String   [VARIABLES] Throwable  cause  NestableDelegate  delegate  boolean  
[BugLab_Wrong_Operator]^if  ( index != 0 )  {^114^^^^^113^119^if  ( index == 0 )  {^[CLASS] NestableError  [METHOD] getMessage [RETURN_TYPE] String   int index [VARIABLES] Throwable  cause  boolean  NestableDelegate  delegate  int  index  
[BugLab_Argument_Swapping]^return index.getMessage ( delegate ) ;^117^^^^^113^119^return delegate.getMessage ( index ) ;^[CLASS] NestableError  [METHOD] getMessage [RETURN_TYPE] String   int index [VARIABLES] Throwable  cause  boolean  NestableDelegate  delegate  int  index  
[BugLab_Argument_Swapping]^return index.getThrowable ( delegate ) ;^132^^^^^131^133^return delegate.getThrowable ( index ) ;^[CLASS] NestableError  [METHOD] getThrowable [RETURN_TYPE] Throwable   int index [VARIABLES] Throwable  cause  boolean  NestableDelegate  delegate  int  index  
[BugLab_Argument_Swapping]^return type.indexOfThrowable ( delegate, 0 ) ;^153^^^^^152^154^return delegate.indexOfThrowable ( type, 0 ) ;^[CLASS] NestableError  [METHOD] indexOfThrowable [RETURN_TYPE] int   Class type [VARIABLES] Throwable  cause  Class  type  boolean  NestableDelegate  delegate  
[BugLab_Wrong_Literal]^return delegate.indexOfThrowable ( type, 1 ) ;^153^^^^^152^154^return delegate.indexOfThrowable ( type, 0 ) ;^[CLASS] NestableError  [METHOD] indexOfThrowable [RETURN_TYPE] int   Class type [VARIABLES] Throwable  cause  Class  type  boolean  NestableDelegate  delegate  
[BugLab_Argument_Swapping]^return type.indexOfThrowable ( delegate, fromIndex ) ;^160^^^^^159^161^return delegate.indexOfThrowable ( type, fromIndex ) ;^[CLASS] NestableError  [METHOD] indexOfThrowable [RETURN_TYPE] int   Class type int fromIndex [VARIABLES] Throwable  cause  Class  type  boolean  NestableDelegate  delegate  int  fromIndex  
[BugLab_Argument_Swapping]^return delegate.indexOfThrowable ( fromIndex, type ) ;^160^^^^^159^161^return delegate.indexOfThrowable ( type, fromIndex ) ;^[CLASS] NestableError  [METHOD] indexOfThrowable [RETURN_TYPE] int   Class type int fromIndex [VARIABLES] Throwable  cause  Class  type  boolean  NestableDelegate  delegate  int  fromIndex  
[BugLab_Argument_Swapping]^return fromIndex.indexOfThrowable ( type, delegate ) ;^160^^^^^159^161^return delegate.indexOfThrowable ( type, fromIndex ) ;^[CLASS] NestableError  [METHOD] indexOfThrowable [RETURN_TYPE] int   Class type int fromIndex [VARIABLES] Throwable  cause  Class  type  boolean  NestableDelegate  delegate  int  fromIndex  
