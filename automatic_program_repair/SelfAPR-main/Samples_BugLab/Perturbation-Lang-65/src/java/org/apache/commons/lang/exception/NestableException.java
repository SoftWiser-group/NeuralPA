[BugLab_Wrong_Operator]^if  ( super.getMessage (  )  == null )  {^161^^^^^160^168^if  ( super.getMessage (  )  != null )  {^[CLASS] NestableException  [METHOD] getMessage [RETURN_TYPE] String   [VARIABLES] Throwable  cause  NestableDelegate  delegate  boolean  
[BugLab_Wrong_Operator]^} else if  ( cause == null )  {^163^^^^^160^168^} else if  ( cause != null )  {^[CLASS] NestableException  [METHOD] getMessage [RETURN_TYPE] String   [VARIABLES] Throwable  cause  NestableDelegate  delegate  boolean  
[BugLab_Wrong_Operator]^if  ( index != 0 )  {^174^^^^^173^179^if  ( index == 0 )  {^[CLASS] NestableException  [METHOD] getMessage [RETURN_TYPE] String   int index [VARIABLES] Throwable  cause  boolean  NestableDelegate  delegate  int  index  
[BugLab_Wrong_Literal]^if  ( index == 1 )  {^174^^^^^173^179^if  ( index == 0 )  {^[CLASS] NestableException  [METHOD] getMessage [RETURN_TYPE] String   int index [VARIABLES] Throwable  cause  boolean  NestableDelegate  delegate  int  index  
[BugLab_Argument_Swapping]^return index.getMessage ( delegate ) ;^177^^^^^173^179^return delegate.getMessage ( index ) ;^[CLASS] NestableException  [METHOD] getMessage [RETURN_TYPE] String   int index [VARIABLES] Throwable  cause  boolean  NestableDelegate  delegate  int  index  
[BugLab_Argument_Swapping]^return index.getThrowable ( delegate ) ;^192^^^^^191^193^return delegate.getThrowable ( index ) ;^[CLASS] NestableException  [METHOD] getThrowable [RETURN_TYPE] Throwable   int index [VARIABLES] Throwable  cause  boolean  NestableDelegate  delegate  int  index  
[BugLab_Argument_Swapping]^return type.indexOfThrowable ( delegate, 0 ) ;^213^^^^^212^214^return delegate.indexOfThrowable ( type, 0 ) ;^[CLASS] NestableException  [METHOD] indexOfThrowable [RETURN_TYPE] int   Class type [VARIABLES] Throwable  cause  Class  type  boolean  NestableDelegate  delegate  
[BugLab_Wrong_Literal]^return delegate.indexOfThrowable ( type, -1 ) ;^213^^^^^212^214^return delegate.indexOfThrowable ( type, 0 ) ;^[CLASS] NestableException  [METHOD] indexOfThrowable [RETURN_TYPE] int   Class type [VARIABLES] Throwable  cause  Class  type  boolean  NestableDelegate  delegate  
[BugLab_Argument_Swapping]^return type.indexOfThrowable ( delegate, fromIndex ) ;^220^^^^^219^221^return delegate.indexOfThrowable ( type, fromIndex ) ;^[CLASS] NestableException  [METHOD] indexOfThrowable [RETURN_TYPE] int   Class type int fromIndex [VARIABLES] Throwable  cause  Class  type  boolean  NestableDelegate  delegate  int  fromIndex  
[BugLab_Argument_Swapping]^return delegate.indexOfThrowable ( fromIndex, type ) ;^220^^^^^219^221^return delegate.indexOfThrowable ( type, fromIndex ) ;^[CLASS] NestableException  [METHOD] indexOfThrowable [RETURN_TYPE] int   Class type int fromIndex [VARIABLES] Throwable  cause  Class  type  boolean  NestableDelegate  delegate  int  fromIndex  
[BugLab_Argument_Swapping]^return fromIndex.indexOfThrowable ( type, delegate ) ;^220^^^^^219^221^return delegate.indexOfThrowable ( type, fromIndex ) ;^[CLASS] NestableException  [METHOD] indexOfThrowable [RETURN_TYPE] int   Class type int fromIndex [VARIABLES] Throwable  cause  Class  type  boolean  NestableDelegate  delegate  int  fromIndex  
