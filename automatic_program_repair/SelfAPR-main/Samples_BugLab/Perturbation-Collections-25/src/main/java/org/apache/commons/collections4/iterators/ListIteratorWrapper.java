[BugLab_Wrong_Operator]^if  ( iterator != null )  {^80^^^^^78^84^if  ( iterator == null )  {^[CLASS] ListIteratorWrapper  [METHOD] <init> [RETURN_TYPE] Iterator)   Iterator<? extends E> iterator [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( iterator  |  ListIterator )  {^97^^^^^96^104^if  ( iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] boolean  removeState  E  obj  Iterator  iterator  List  list  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Argument_Swapping]^if  ( iterator == wrappedIteratorIndex || currentIndex instanceof ListIterator )  {^112^^^^^111^116^if  ( currentIndex == wrappedIteratorIndex || iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  wrappedIteratorIndex  
[BugLab_Argument_Swapping]^if  ( wrappedIteratorIndex == currentIndex || iterator instanceof ListIterator )  {^112^^^^^111^116^if  ( currentIndex == wrappedIteratorIndex || iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  wrappedIteratorIndex  
[BugLab_Argument_Swapping]^if  ( currentIndex == iterator || wrappedIteratorIndex instanceof ListIterator )  {^112^^^^^111^116^if  ( currentIndex == wrappedIteratorIndex || iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( currentIndex == wrappedIteratorIndex && iterator instanceof ListIterator )  {^112^^^^^111^116^if  ( currentIndex == wrappedIteratorIndex || iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( currentIndex >= wrappedIteratorIndex || iterator instanceof ListIterator )  {^112^^^^^111^116^if  ( currentIndex == wrappedIteratorIndex || iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( currentIndex == wrappedIteratorIndex || iterator  <  ListIterator )  {^112^^^^^111^116^if  ( currentIndex == wrappedIteratorIndex || iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^return this.hasNext (  ) ;^113^^^^^111^116^return iterator.hasNext (  ) ;^[CLASS] ListIteratorWrapper  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Literal]^return false;^115^^^^^111^116^return true;^[CLASS] ListIteratorWrapper  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( iterator  <<  ListIterator )  {^124^^^^^123^129^if  ( iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] hasPrevious [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^return wrappedIteratorIndex > 0;^128^^^^^123^129^return currentIndex > 0;^[CLASS] ListIteratorWrapper  [METHOD] hasPrevious [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^return currentIndex < 0;^128^^^^^123^129^return currentIndex > 0;^[CLASS] ListIteratorWrapper  [METHOD] hasPrevious [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Literal]^return currentIndex > 1;^128^^^^^123^129^return currentIndex > 0;^[CLASS] ListIteratorWrapper  [METHOD] hasPrevious [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( iterator  <<  ListIterator )  {^138^^^^^137^153^if  ( iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^return 4.next (  ) ;^139^^^^^137^153^return iterator.next (  ) ;^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^if  ( wrappedIteratorIndex < wrappedIteratorIndex )  {^142^^^^^137^153^if  ( currentIndex < wrappedIteratorIndex )  {^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Argument_Swapping]^if  ( wrappedIteratorIndex < currentIndex )  {^142^^^^^137^153^if  ( currentIndex < wrappedIteratorIndex )  {^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( currentIndex <= wrappedIteratorIndex )  {^142^^^^^137^153^if  ( currentIndex < wrappedIteratorIndex )  {^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^return list.get ( wrappedIteratorIndex - 1 ) ;^144^^^^^137^153^return list.get ( currentIndex - 1 ) ;^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Argument_Swapping]^return currentIndex.get ( list - 1 ) ;^144^^^^^137^153^return list.get ( currentIndex - 1 ) ;^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^return list.get ( currentIndex  <  1 ) ;^144^^^^^137^153^return list.get ( currentIndex - 1 ) ;^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Literal]^return list.get ( currentIndex - currentIndex ) ;^144^^^^^137^153^return list.get ( currentIndex - 1 ) ;^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^return list.get ( currentIndex  ^  1 ) ;^144^^^^^137^153^return list.get ( currentIndex - 1 ) ;^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Literal]^return list.get ( currentIndex  ) ;^144^^^^^137^153^return list.get ( currentIndex - 1 ) ;^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^return list.get ( currentIndex  >>  1 ) ;^144^^^^^137^153^return list.get ( currentIndex - 1 ) ;^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Literal]^removeState = false;^151^^^^^137^153^removeState = true;^[CLASS] ListIteratorWrapper  [METHOD] next [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  E  retval  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( iterator  &  ListIterator )  {^161^^^^^160^166^if  ( iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] nextIndex [RETURN_TYPE] int   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^return 0.nextIndex (  ) ;^163^^^^^160^166^return li.nextIndex (  ) ;^[CLASS] ListIteratorWrapper  [METHOD] nextIndex [RETURN_TYPE] int   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^return wrappedIteratorIndex;^165^^^^^160^166^return currentIndex;^[CLASS] ListIteratorWrapper  [METHOD] nextIndex [RETURN_TYPE] int   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( iterator  >=  ListIterator )  {^175^^^^^174^186^if  ( iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^if  ( wrappedIteratorIndex == 0 )  {^181^^^^^174^186^if  ( currentIndex == 0 )  {^[CLASS] ListIteratorWrapper  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( currentIndex <= 0 )  {^181^^^^^174^186^if  ( currentIndex == 0 )  {^[CLASS] ListIteratorWrapper  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Literal]^if  ( currentIndex == wrappedIteratorIndex )  {^181^^^^^174^186^if  ( currentIndex == 0 )  {^[CLASS] ListIteratorWrapper  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Argument_Swapping]^removeState = currentIndex == wrappedIteratorIndex;^184^^^^^174^186^removeState = wrappedIteratorIndex == currentIndex;^[CLASS] ListIteratorWrapper  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^removeState = wrappedIteratorIndex != currentIndex;^184^^^^^174^186^removeState = wrappedIteratorIndex == currentIndex;^[CLASS] ListIteratorWrapper  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( iterator  &  ListIterator )  {^194^^^^^193^199^if  ( iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] previousIndex [RETURN_TYPE] int   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^return wrappedIteratorIndex - 1;^198^^^^^193^199^return currentIndex - 1;^[CLASS] ListIteratorWrapper  [METHOD] previousIndex [RETURN_TYPE] int   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^return currentIndex  <  1;^198^^^^^193^199^return currentIndex - 1;^[CLASS] ListIteratorWrapper  [METHOD] previousIndex [RETURN_TYPE] int   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Literal]^return currentIndex ;^198^^^^^193^199^return currentIndex - 1;^[CLASS] ListIteratorWrapper  [METHOD] previousIndex [RETURN_TYPE] int   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( iterator  >=  ListIterator )  {^207^^^^^206^223^if  ( iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^if  ( removeIndex == wrappedIteratorIndex )  {^212^^^^^206^223^if  ( currentIndex == wrappedIteratorIndex )  {^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^if  ( currentIndex == removeIndex )  {^212^^^^^206^223^if  ( currentIndex == wrappedIteratorIndex )  {^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Argument_Swapping]^if  ( wrappedIteratorIndex == currentIndex )  {^212^^^^^206^223^if  ( currentIndex == wrappedIteratorIndex )  {^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( currentIndex != wrappedIteratorIndex )  {^212^^^^^206^223^if  ( currentIndex == wrappedIteratorIndex )  {^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^if  ( !removeState || currentIndex - currentIndex > 1 )  {^215^^^^^206^223^if  ( !removeState || wrappedIteratorIndex - currentIndex > 1 )  {^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^if  ( !removeState || wrappedIteratorIndex - wrappedIteratorIndex > 1 )  {^215^^^^^206^223^if  ( !removeState || wrappedIteratorIndex - currentIndex > 1 )  {^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Argument_Swapping]^if  ( !removeState || currentIndex - wrappedIteratorIndex > 1 )  {^215^^^^^206^223^if  ( !removeState || wrappedIteratorIndex - currentIndex > 1 )  {^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( !removeState && wrappedIteratorIndex - currentIndex > 1 )  {^215^^^^^206^223^if  ( !removeState || wrappedIteratorIndex - currentIndex > 1 )  {^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( !removeState || wrappedIteratorIndex - currentIndex >= 1 )  {^215^^^^^206^223^if  ( !removeState || wrappedIteratorIndex - currentIndex > 1 )  {^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( !removeState || wrappedIteratorIndex  !=  currentIndex > 1 )  {^215^^^^^206^223^if  ( !removeState || wrappedIteratorIndex - currentIndex > 1 )  {^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Literal]^if  ( !removeState || wrappedIteratorIndex - currentIndex >  )  {^215^^^^^206^223^if  ( !removeState || wrappedIteratorIndex - currentIndex > 1 )  {^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^list.remove ( wrappedIteratorIndex ) ;^219^^^^^206^223^list.remove ( removeIndex ) ;^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^currentIndex = wrappedIteratorIndex;^220^^^^^206^223^currentIndex = removeIndex;^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Literal]^removeState = true;^222^^^^^206^223^removeState = false;^[CLASS] ListIteratorWrapper  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( iterator  >=  ListIterator )  {^234^^^^^233^241^if  ( iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] boolean  removeState  E  obj  Iterator  iterator  List  list  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^while  ( li.previousIndex (  )  == 0 )  {^254^^^^^251^260^while  ( li.previousIndex (  )  >= 0 )  {^[CLASS] ListIteratorWrapper  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Literal]^while  ( li.previousIndex (  )  >= -1 )  {^254^^^^^251^260^while  ( li.previousIndex (  )  >= 0 )  {^[CLASS] ListIteratorWrapper  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Variable_Misuse]^if  ( null instanceof ListIterator )  {^252^^^^^251^260^if  ( iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^if  ( iterator  <<  ListIterator )  {^252^^^^^251^260^if  ( iterator instanceof ListIterator )  {^[CLASS] ListIteratorWrapper  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Operator]^while  ( li.previousIndex (  )  > 0 )  {^254^^^^^251^260^while  ( li.previousIndex (  )  >= 0 )  {^[CLASS] ListIteratorWrapper  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Literal]^while  ( li.previousIndex (  )  >= removeIndex )  {^254^^^^^251^260^while  ( li.previousIndex (  )  >= 0 )  {^[CLASS] ListIteratorWrapper  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  removeIndex  wrappedIteratorIndex  
[BugLab_Wrong_Literal]^currentIndex = currentIndex;^259^^^^^251^260^currentIndex = 0;^[CLASS] ListIteratorWrapper  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] Iterator  iterator  List  list  boolean  removeState  String  CANNOT_REMOVE_MESSAGE  UNSUPPORTED_OPERATION_MESSAGE  ListIterator  li  int  currentIndex  removeIndex  wrappedIteratorIndex  