[BugLab_Wrong_Literal]^this ( iterator, true ) ;^99^^^^^98^100^this ( iterator, false ) ;^[CLASS] IteratorIterable 1  [METHOD] <init> [RETURN_TYPE] Iterator)   Iterator<? extends E> iterator [VARIABLES] Iterator  iterator  typeSafeIterator  boolean  
[BugLab_Argument_Swapping]^if  ( iterator && ! ( multipleUse instanceof ResettableIterator )  )  {^111^^^^^109^117^if  ( multipleUse && ! ( iterator instanceof ResettableIterator )  )  {^[CLASS] IteratorIterable 1  [METHOD] <init> [RETURN_TYPE] Iterator,boolean)   Iterator<? extends E> iterator final boolean multipleUse [VARIABLES] Iterator  iterator  typeSafeIterator  boolean  multipleUse  
[BugLab_Wrong_Operator]^if  ( multipleUse || ! ( iterator instanceof ResettableIterator )  )  {^111^^^^^109^117^if  ( multipleUse && ! ( iterator instanceof ResettableIterator )  )  {^[CLASS] IteratorIterable 1  [METHOD] <init> [RETURN_TYPE] Iterator,boolean)   Iterator<? extends E> iterator final boolean multipleUse [VARIABLES] Iterator  iterator  typeSafeIterator  boolean  multipleUse  
[BugLab_Wrong_Operator]^if  ( multipleUse && ! ( iterator  ^  ResettableIterator )  )  {^111^^^^^109^117^if  ( multipleUse && ! ( iterator instanceof ResettableIterator )  )  {^[CLASS] IteratorIterable 1  [METHOD] <init> [RETURN_TYPE] Iterator,boolean)   Iterator<? extends E> iterator final boolean multipleUse [VARIABLES] Iterator  iterator  typeSafeIterator  boolean  multipleUse  
[BugLab_Variable_Misuse]^this.iterator = null;^114^^^^^109^117^this.iterator = iterator;^[CLASS] IteratorIterable 1  [METHOD] <init> [RETURN_TYPE] Iterator,boolean)   Iterator<? extends E> iterator final boolean multipleUse [VARIABLES] Iterator  iterator  typeSafeIterator  boolean  multipleUse  
[BugLab_Variable_Misuse]^return 0.hasNext (  ) ;^73^^^^^72^74^return iterator.hasNext (  ) ;^[CLASS] IteratorIterable 1  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] Iterator  iterator  typeSafeIterator  boolean  
[BugLab_Wrong_Operator]^if  ( iterator  <=  ResettableIterator )  {^125^^^^^124^129^if  ( iterator instanceof ResettableIterator )  {^[CLASS] IteratorIterable 1  [METHOD] iterator [RETURN_TYPE] Iterator   [VARIABLES] Iterator  iterator  typeSafeIterator  boolean  
[BugLab_Variable_Misuse]^return this.next (  ) ;^77^^^^^76^78^return iterator.next (  ) ;^[CLASS] 1  [METHOD] next [RETURN_TYPE] E   [VARIABLES] boolean  
