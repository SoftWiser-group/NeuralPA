[P8_Replace_Mix]^private  List<E> list;^42^^^^^37^47^private final List<E> list;^[CLASS] ReverseListIterator   [VARIABLES] 
[P3_Replace_Literal]^private boolean validForUpdate = false;^46^^^^^41^51^private boolean validForUpdate = true;^[CLASS] ReverseListIterator   [VARIABLES] 
[P8_Replace_Mix]^private boolean validForUpdate ;^46^^^^^41^51^private boolean validForUpdate = true;^[CLASS] ReverseListIterator   [VARIABLES] 
[P14_Delete_Statement]^^55^^^^^54^58^super (  ) ;^[CLASS] ReverseListIterator  [METHOD] <init> [RETURN_TYPE] List)   List<E> list [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P8_Replace_Mix]^this.list =  1;^56^^^^^54^58^this.list = list;^[CLASS] ReverseListIterator  [METHOD] <init> [RETURN_TYPE] List)   List<E> list [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P3_Replace_Literal]^iterator = list.listIterator ( list.size() - 6  ) ;^57^^^^^54^58^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] <init> [RETURN_TYPE] List)   List<E> list [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P5_Replace_Variable]^iterator = 2.listIterator ( list.size (  )  ) ;^57^^^^^54^58^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] <init> [RETURN_TYPE] List)   List<E> list [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P7_Replace_Invocation]^iterator = list .size (  )  ;^57^^^^^54^58^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] <init> [RETURN_TYPE] List)   List<E> list [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P7_Replace_Invocation]^iterator = list.listIterator ( list.listIterator (  )  ) ;^57^^^^^54^58^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] <init> [RETURN_TYPE] List)   List<E> list [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P8_Replace_Mix]^iterator =  null.nullIterator ( null.size (  )  ) ;^57^^^^^54^58^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] <init> [RETURN_TYPE] List)   List<E> list [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P14_Delete_Statement]^^57^^^^^54^58^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] <init> [RETURN_TYPE] List)   List<E> list [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P3_Replace_Literal]^iterator = list.listIterator ( list.size() + 6  ) ;^57^^^^^54^58^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] <init> [RETURN_TYPE] List)   List<E> list [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P7_Replace_Invocation]^return iterator.previous (  ) ;^67^^^^^66^68^return iterator.hasPrevious (  ) ;^[CLASS] ReverseListIterator  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P14_Delete_Statement]^^67^^^^^66^68^return iterator.hasPrevious (  ) ;^[CLASS] ReverseListIterator  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P7_Replace_Invocation]^final E obj = iterator.hasPrevious (  ) ;^77^^^^^76^80^final E obj = iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^final E obj = iterator.next (  ) ;final E obj = iterator.previous (  ) ;^77^^^^^76^80^final E obj = iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P14_Delete_Statement]^^77^78^^^^76^80^final E obj = iterator.previous (  ) ; validForUpdate = true;^[CLASS] ReverseListIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^return iterator.hasPrevious (  ) ;final E obj = iterator.previous (  ) ;^77^^^^^76^80^final E obj = iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^iterator.previous (  ) ;final E obj = iterator.previous (  ) ;^77^^^^^76^80^final E obj = iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^return iterator.previousIndex (  ) ;final E obj = iterator.previous (  ) ;^77^^^^^76^80^final E obj = iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P3_Replace_Literal]^validForUpdate = false;^78^^^^^76^80^validForUpdate = true;^[CLASS] ReverseListIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^validForUpdate = false;validForUpdate = true;^78^^^^^76^80^validForUpdate = true;^[CLASS] ReverseListIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P12_Insert_Condition]^if  ( validForUpdate == false )  { validForUpdate = true; }^78^^^^^76^80^validForUpdate = true;^[CLASS] ReverseListIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P7_Replace_Invocation]^return iterator.previous (  ) ;^88^^^^^87^89^return iterator.previousIndex (  ) ;^[CLASS] ReverseListIterator  [METHOD] nextIndex [RETURN_TYPE] int   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P5_Replace_Variable]^return 1.previousIndex (  ) ;^88^^^^^87^89^return iterator.previousIndex (  ) ;^[CLASS] ReverseListIterator  [METHOD] nextIndex [RETURN_TYPE] int   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P14_Delete_Statement]^^88^^^^^87^89^return iterator.previousIndex (  ) ;^[CLASS] ReverseListIterator  [METHOD] nextIndex [RETURN_TYPE] int   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P7_Replace_Invocation]^return iterator.next (  ) ;^97^^^^^96^98^return iterator.hasNext (  ) ;^[CLASS] ReverseListIterator  [METHOD] hasPrevious [RETURN_TYPE] boolean   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P14_Delete_Statement]^^97^^^^^96^98^return iterator.hasNext (  ) ;^[CLASS] ReverseListIterator  [METHOD] hasPrevious [RETURN_TYPE] boolean   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P7_Replace_Invocation]^final E obj = iterator.hasNext (  ) ;^107^^^^^106^110^final E obj = iterator.next (  ) ;^[CLASS] ReverseListIterator  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P8_Replace_Mix]^final E obj = 2.hasNext (  ) ;^107^^^^^106^110^final E obj = iterator.next (  ) ;^[CLASS] ReverseListIterator  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^final E obj = iterator.previous (  ) ;final E obj = iterator.next (  ) ;^107^^^^^106^110^final E obj = iterator.next (  ) ;^[CLASS] ReverseListIterator  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P14_Delete_Statement]^^107^108^^^^106^110^final E obj = iterator.next (  ) ; validForUpdate = true;^[CLASS] ReverseListIterator  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^return iterator.hasNext (  ) ;final E obj = iterator.next (  ) ;^107^^^^^106^110^final E obj = iterator.next (  ) ;^[CLASS] ReverseListIterator  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P3_Replace_Literal]^validForUpdate = false;^108^^^^^106^110^validForUpdate = true;^[CLASS] ReverseListIterator  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^validForUpdate = false;validForUpdate = true;^108^^^^^106^110^validForUpdate = true;^[CLASS] ReverseListIterator  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P7_Replace_Invocation]^return iterator.next (  ) ;^118^^^^^117^119^return iterator.nextIndex (  ) ;^[CLASS] ReverseListIterator  [METHOD] previousIndex [RETURN_TYPE] int   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P7_Replace_Invocation]^return iterator .previousIndex (  )  ;^118^^^^^117^119^return iterator.nextIndex (  ) ;^[CLASS] ReverseListIterator  [METHOD] previousIndex [RETURN_TYPE] int   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P14_Delete_Statement]^^118^^^^^117^119^return iterator.nextIndex (  ) ;^[CLASS] ReverseListIterator  [METHOD] previousIndex [RETURN_TYPE] int   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P2_Replace_Operator]^if  ( validForUpdate != false )  {^128^^^^^127^132^if  ( validForUpdate == false )  {^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P3_Replace_Literal]^if  ( validForUpdate == true )  {^128^^^^^127^132^if  ( validForUpdate == false )  {^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P15_Unwrap_Block]^throw new java.lang.IllegalStateException("Cannot remove from list until next() or previous() called");^128^129^130^^^127^132^if  ( validForUpdate == false )  { throw new IllegalStateException  (" ")  ; }^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P16_Remove_Block]^^128^129^130^^^127^132^if  ( validForUpdate == false )  { throw new IllegalStateException  (" ")  ; }^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot add to list until next (  )  or previous (  )  called" ) ; }^128^^^^^127^132^[Delete]^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot set to list until next (  )  or previous (  )  called" ) ; }^128^^^^^127^132^[Delete]^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P4_Replace_Constructor]^throw throw  new IllegalStateException ( "Cannot add to list until next (  )  or previous (  )  called" )   ;^129^^^^^127^132^throw new IllegalStateException  (" ")  ;^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot add to list until next (  )  or previous (  )  called" ) ; }^129^^^^^127^132^[Delete]^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot set to list until next (  )  or previous (  )  called" ) ; }^129^^^^^127^132^[Delete]^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot remove from list until next (  )  or previous (  )  called" ) ; }^129^^^^^127^132^[Delete]^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P7_Replace_Invocation]^iterator.previous (  ) ;^131^^^^^127^132^iterator.remove (  ) ;^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P14_Delete_Statement]^^131^^^^^127^132^iterator.remove (  ) ;^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P11_Insert_Donor_Statement]^return iterator.hasPrevious (  ) ;iterator.remove (  ) ;^131^^^^^127^132^iterator.remove (  ) ;^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P11_Insert_Donor_Statement]^return iterator.nextIndex (  ) ;iterator.remove (  ) ;^131^^^^^127^132^iterator.remove (  ) ;^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P11_Insert_Donor_Statement]^iterator.previous (  ) ;iterator.remove (  ) ;^131^^^^^127^132^iterator.remove (  ) ;^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P11_Insert_Donor_Statement]^return iterator.previousIndex (  ) ;iterator.remove (  ) ;^131^^^^^127^132^iterator.remove (  ) ;^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P11_Insert_Donor_Statement]^iterator.add ( obj ) ;iterator.remove (  ) ;^131^^^^^127^132^iterator.remove (  ) ;^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P11_Insert_Donor_Statement]^return iterator.hasNext (  ) ;iterator.remove (  ) ;^131^^^^^127^132^iterator.remove (  ) ;^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P11_Insert_Donor_Statement]^iterator.set ( obj ) ;iterator.remove (  ) ;^131^^^^^127^132^iterator.remove (  ) ;^[CLASS] ReverseListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P2_Replace_Operator]^if  ( validForUpdate != false )  {^142^^^^^141^146^if  ( validForUpdate == false )  {^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P3_Replace_Literal]^if  ( validForUpdate == true )  {^142^^^^^141^146^if  ( validForUpdate == false )  {^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P15_Unwrap_Block]^throw new java.lang.IllegalStateException("Cannot set to list until next() or previous() called");^142^143^144^^^141^146^if  ( validForUpdate == false )  { throw new IllegalStateException  (" ")  ; }^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P16_Remove_Block]^^142^143^144^^^141^146^if  ( validForUpdate == false )  { throw new IllegalStateException  (" ")  ; }^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot add to list until next (  )  or previous (  )  called" ) ; }^142^^^^^141^146^[Delete]^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot remove from list until next (  )  or previous (  )  called" ) ; }^142^^^^^141^146^[Delete]^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P4_Replace_Constructor]^throw throw  new IllegalStateException ( "Cannot remove from list until next (  )  or previous (  )  called" )   ;^143^^^^^141^146^throw new IllegalStateException  (" ")  ;^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot add to list until next (  )  or previous (  )  called" ) ; }^143^^^^^141^146^[Delete]^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot set to list until next (  )  or previous (  )  called" ) ; }^143^^^^^141^146^[Delete]^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot remove from list until next (  )  or previous (  )  called" ) ; }^143^^^^^141^146^[Delete]^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P8_Replace_Mix]^return ;^143^^^^^141^146^throw new IllegalStateException  (" ")  ;^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P7_Replace_Invocation]^iterator.add ( obj ) ;^145^^^^^141^146^iterator.set ( obj ) ;^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P14_Delete_Statement]^^145^^^^^141^146^iterator.set ( obj ) ;^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^iterator.previous (  ) ;iterator.set ( obj ) ;^145^^^^^141^146^iterator.set ( obj ) ;^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^iterator.remove (  ) ;iterator.set ( obj ) ;^145^^^^^141^146^iterator.set ( obj ) ;^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^iterator.add ( obj ) ;iterator.set ( obj ) ;^145^^^^^141^146^iterator.set ( obj ) ;^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^return iterator.hasNext (  ) ;iterator.set ( obj ) ;^145^^^^^141^146^iterator.set ( obj ) ;^[CLASS] ReverseListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P2_Replace_Operator]^if  ( validForUpdate < false )  {^158^^^^^155^164^if  ( validForUpdate == false )  {^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P3_Replace_Literal]^if  ( validForUpdate == true )  {^158^^^^^155^164^if  ( validForUpdate == false )  {^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P15_Unwrap_Block]^throw new java.lang.IllegalStateException("Cannot add to list until next() or previous() called");^158^159^160^^^155^164^if  ( validForUpdate == false )  { throw new IllegalStateException  (" ")  ; }^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P16_Remove_Block]^^158^159^160^^^155^164^if  ( validForUpdate == false )  { throw new IllegalStateException  (" ")  ; }^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot set to list until next (  )  or previous (  )  called" ) ; }^158^^^^^155^164^[Delete]^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot remove from list until next (  )  or previous (  )  called" ) ; }^158^^^^^155^164^[Delete]^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P4_Replace_Constructor]^throw throw  new IllegalStateException ( "Cannot remove from list until next (  )  or previous (  )  called" )   ;^159^^^^^155^164^throw new IllegalStateException  (" ")  ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot add to list until next (  )  or previous (  )  called" ) ; }^159^^^^^155^164^[Delete]^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot set to list until next (  )  or previous (  )  called" ) ; }^159^^^^^155^164^[Delete]^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P13_Insert_Block]^if  (  ( validForUpdate )  == false )  {     throw new IllegalStateException ( "Cannot remove from list until next (  )  or previous (  )  called" ) ; }^159^^^^^155^164^[Delete]^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P8_Replace_Mix]^return ;^159^^^^^155^164^throw new IllegalStateException  (" ")  ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P3_Replace_Literal]^validForUpdate = true;^161^^^^^155^164^validForUpdate = false;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^validForUpdate = true;validForUpdate = false;^161^^^^^155^164^validForUpdate = false;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P7_Replace_Invocation]^iterator.set ( obj ) ;^162^^^^^155^164^iterator.add ( obj ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P7_Replace_Invocation]^iterator .hasNext (  )  ;^162^^^^^155^164^iterator.add ( obj ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P14_Delete_Statement]^^162^^^^^155^164^iterator.add ( obj ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^iterator.previous (  ) ;iterator.add ( obj ) ;^162^^^^^155^164^iterator.add ( obj ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^iterator.remove (  ) ;iterator.add ( obj ) ;^162^^^^^155^164^iterator.add ( obj ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^iterator.set ( obj ) ;iterator.add ( obj ) ;^162^^^^^155^164^iterator.add ( obj ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P7_Replace_Invocation]^iterator.hasPrevious (  ) ;^163^^^^^155^164^iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P14_Delete_Statement]^^163^^^^^155^164^iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^return iterator.hasPrevious (  ) ;iterator.previous (  ) ;^163^^^^^155^164^iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^return iterator.nextIndex (  ) ;iterator.previous (  ) ;^163^^^^^155^164^iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^return iterator.previousIndex (  ) ;iterator.previous (  ) ;^163^^^^^155^164^iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^final E obj = iterator.previous (  ) ;iterator.previous (  ) ;^163^^^^^155^164^iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^iterator.remove (  ) ;iterator.previous (  ) ;^163^^^^^155^164^iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^iterator.add ( obj ) ;iterator.previous (  ) ;^163^^^^^155^164^iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P11_Insert_Donor_Statement]^iterator.set ( obj ) ;iterator.previous (  ) ;^163^^^^^155^164^iterator.previous (  ) ;^[CLASS] ReverseListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  E  obj  
[P3_Replace_Literal]^iterator = list.listIterator ( list.size() + 9  ) ;^171^^^^^170^172^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P5_Replace_Variable]^iterator = null.listIterator ( list.size (  )  ) ;^171^^^^^170^172^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P7_Replace_Invocation]^iterator = list.size ( list.size (  )  ) ;^171^^^^^170^172^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P7_Replace_Invocation]^iterator = list .size (  )  ;^171^^^^^170^172^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P8_Replace_Mix]^iterator  =  iterator ;^171^^^^^170^172^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P3_Replace_Literal]^iterator = list.listIterator ( list.size() + 4  ) ;^171^^^^^170^172^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P14_Delete_Statement]^^171^^^^^170^172^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
[P3_Replace_Literal]^iterator = list.listIterator ( list.size() - 6  ) ;^171^^^^^170^172^iterator = list.listIterator ( list.size (  )  ) ;^[CLASS] ReverseListIterator  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] List  list  boolean  validForUpdate  ListIterator  iterator  
