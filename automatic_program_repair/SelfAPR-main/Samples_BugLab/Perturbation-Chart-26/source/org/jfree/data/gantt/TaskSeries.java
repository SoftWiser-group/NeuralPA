[BugLab_Wrong_Operator]^if  ( task != null )  {^83^^^^^82^88^if  ( task == null )  {^[CLASS] TaskSeries  [METHOD] add [RETURN_TYPE] void   Task task [VARIABLES] List  tasks  Task  task  boolean  
[BugLab_Variable_Misuse]^return tasks.size (  ) ;^118^^^^^117^119^return this.tasks.size (  ) ;^[CLASS] TaskSeries  [METHOD] getItemCount [RETURN_TYPE] int   [VARIABLES] List  tasks  boolean  
[BugLab_Variable_Misuse]^return  ( Task )  tasks.get ( index ) ;^129^^^^^128^130^return  ( Task )  this.tasks.get ( index ) ;^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   int index [VARIABLES] List  tasks  int  index  boolean  
[BugLab_Argument_Swapping]^return  ( Task )  index.get ( this.tasks ) ;^129^^^^^128^130^return  ( Task )  this.tasks.get ( index ) ;^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   int index [VARIABLES] List  tasks  int  index  boolean  
[BugLab_Variable_Misuse]^int count = tasks.size (  ) ;^141^^^^^139^150^int count = this.tasks.size (  ) ;^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Variable_Misuse]^if  ( result.getDescription (  ) .equals ( description )  )  {^144^^^^^139^150^if  ( t.getDescription (  ) .equals ( description )  )  {^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Argument_Swapping]^if  ( description.getDescription (  ) .equals ( t )  )  {^144^^^^^139^150^if  ( t.getDescription (  ) .equals ( description )  )  {^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Variable_Misuse]^result = result;^145^^^^^139^150^result = t;^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Variable_Misuse]^for  ( countnt i = 0; i < count; i++ )  {^142^^^^^139^150^for  ( int i = 0; i < count; i++ )  {^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Variable_Misuse]^for  ( inresult i = 0; i < count; i++ )  {^142^^^^^139^150^for  ( int i = 0; i < count; i++ )  {^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Argument_Swapping]^for  ( countnt i = 0; i < i; i++ )  {^142^^^^^139^150^for  ( int i = 0; i < count; i++ )  {^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i == count; i++ )  {^142^^^^^139^150^for  ( int i = 0; i < count; i++ )  {^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Variable_Misuse]^Task t =  ( Task )  this.tasks.get ( count ) ;^143^^^^^139^150^Task t =  ( Task )  this.tasks.get ( i ) ;^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Variable_Misuse]^Task t =  ( Task )  tasks.get ( i ) ;^143^^^^^139^150^Task t =  ( Task )  this.tasks.get ( i ) ;^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Argument_Swapping]^Task t =  ( Task )  i.get ( this.tasks ) ;^143^^^^^139^150^Task t =  ( Task )  this.tasks.get ( i ) ;^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Wrong_Literal]^for  ( int i = -1; i < count; i++ )  {^142^^^^^139^150^for  ( int i = 0; i < count; i++ )  {^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Variable_Misuse]^return t;^149^^^^^139^150^return result;^[CLASS] TaskSeries  [METHOD] get [RETURN_TYPE] Task   String description [VARIABLES] List  tasks  Task  result  t  String  description  boolean  int  count  i  
[BugLab_Variable_Misuse]^return Collections.unmodifiableList ( tasks ) ;^158^^^^^157^159^return Collections.unmodifiableList ( this.tasks ) ;^[CLASS] TaskSeries  [METHOD] getTasks [RETURN_TYPE] List   [VARIABLES] List  tasks  boolean  
[BugLab_Wrong_Operator]^if  ( obj != this )  {^169^^^^^168^183^if  ( obj == this )  {^[CLASS] TaskSeries  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  tasks  Object  obj  boolean  TaskSeries  that  
[BugLab_Wrong_Literal]^return false;^170^^^^^168^183^return true;^[CLASS] TaskSeries  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  tasks  Object  obj  boolean  TaskSeries  that  
[BugLab_Wrong_Operator]^if  ( ! ( obj  ==  TaskSeries )  )  {^172^^^^^168^183^if  ( ! ( obj instanceof TaskSeries )  )  {^[CLASS] TaskSeries  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  tasks  Object  obj  boolean  TaskSeries  that  
[BugLab_Wrong_Literal]^return true;^173^^^^^168^183^return false;^[CLASS] TaskSeries  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  tasks  Object  obj  boolean  TaskSeries  that  
[BugLab_Wrong_Literal]^return true;^176^^^^^168^183^return false;^[CLASS] TaskSeries  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  tasks  Object  obj  boolean  TaskSeries  that  
[BugLab_Variable_Misuse]^if  ( !this.tasks.equals ( tasks )  )  {^179^^^^^168^183^if  ( !this.tasks.equals ( that.tasks )  )  {^[CLASS] TaskSeries  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  tasks  Object  obj  boolean  TaskSeries  that  
[BugLab_Argument_Swapping]^if  ( !this.tasks.equals ( that.tasks.tasks )  )  {^179^^^^^168^183^if  ( !this.tasks.equals ( that.tasks )  )  {^[CLASS] TaskSeries  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  tasks  Object  obj  boolean  TaskSeries  that  
[BugLab_Argument_Swapping]^if  ( !this.tasks.equals ( that )  )  {^179^^^^^168^183^if  ( !this.tasks.equals ( that.tasks )  )  {^[CLASS] TaskSeries  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  tasks  Object  obj  boolean  TaskSeries  that  
[BugLab_Wrong_Literal]^return true;^180^^^^^168^183^return false;^[CLASS] TaskSeries  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  tasks  Object  obj  boolean  TaskSeries  that  
[BugLab_Wrong_Literal]^return false;^182^^^^^168^183^return true;^[CLASS] TaskSeries  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  tasks  Object  obj  boolean  TaskSeries  that  
