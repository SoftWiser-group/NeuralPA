[BugLab_Wrong_Literal]^private boolean isLocked = true;^69^^^^^64^74^private boolean isLocked = false;^[CLASS] IteratorChain   [VARIABLES] 
[BugLab_Wrong_Operator]^if  ( iterator != null )  {^161^^^^^159^165^if  ( iterator == null )  {^[CLASS] IteratorChain  [METHOD] addIterator [RETURN_TYPE] void   Iterator<? extends E> iterator [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Wrong_Operator]^if  ( isLocked != true )  {^191^^^^^190^195^if  ( isLocked == true )  {^[CLASS] IteratorChain  [METHOD] checkLocked [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Wrong_Literal]^if  ( isLocked == false )  {^191^^^^^190^195^if  ( isLocked == true )  {^[CLASS] IteratorChain  [METHOD] checkLocked [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Wrong_Operator]^if  ( isLocked != false )  {^202^^^^^201^205^if  ( isLocked == false )  {^[CLASS] IteratorChain  [METHOD] lockChain [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Wrong_Literal]^if  ( isLocked == true )  {^202^^^^^201^205^if  ( isLocked == false )  {^[CLASS] IteratorChain  [METHOD] lockChain [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Wrong_Literal]^isLocked = false;^203^^^^^201^205^isLocked = true;^[CLASS] IteratorChain  [METHOD] lockChain [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Wrong_Operator]^if  ( currentIterator != null )  {^212^^^^^211^226^if  ( currentIterator == null )  {^[CLASS] IteratorChain  [METHOD] updateCurrentIterator [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Variable_Misuse]^currentIterator = null.remove (  ) ;^216^^^^^211^226^currentIterator = iteratorChain.remove (  ) ;^[CLASS] IteratorChain  [METHOD] updateCurrentIterator [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Variable_Misuse]^lastUsedIterator = 1;^220^^^^^211^226^lastUsedIterator = currentIterator;^[CLASS] IteratorChain  [METHOD] updateCurrentIterator [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Variable_Misuse]^if  ( 2.isEmpty (  )  )  {^213^^^^^211^226^if  ( iteratorChain.isEmpty (  )  )  {^[CLASS] IteratorChain  [METHOD] updateCurrentIterator [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Variable_Misuse]^while  ( this.hasNext (  )  == false && !iteratorChain.isEmpty (  )  )  {^223^^^^^211^226^while  ( currentIterator.hasNext (  )  == false && !iteratorChain.isEmpty (  )  )  {^[CLASS] IteratorChain  [METHOD] updateCurrentIterator [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Wrong_Operator]^while  ( currentIterator.hasNext (  )  == false || !iteratorChain.isEmpty (  )  )  {^223^^^^^211^226^while  ( currentIterator.hasNext (  )  == false && !iteratorChain.isEmpty (  )  )  {^[CLASS] IteratorChain  [METHOD] updateCurrentIterator [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Wrong_Operator]^while  ( currentIterator.hasNext (  )  >= false && !iteratorChain.isEmpty (  )  )  {^223^^^^^211^226^while  ( currentIterator.hasNext (  )  == false && !iteratorChain.isEmpty (  )  )  {^[CLASS] IteratorChain  [METHOD] updateCurrentIterator [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Wrong_Literal]^while  ( currentIterator.hasNext (  )  == true && !iteratorChain.isEmpty (  )  )  {^223^^^^^211^226^while  ( currentIterator.hasNext (  )  == false && !iteratorChain.isEmpty (  )  )  {^[CLASS] IteratorChain  [METHOD] updateCurrentIterator [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Variable_Misuse]^currentIterator = null.remove (  ) ;^224^^^^^211^226^currentIterator = iteratorChain.remove (  ) ;^[CLASS] IteratorChain  [METHOD] updateCurrentIterator [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
[BugLab_Wrong_Operator]^if  ( currentIterator != null )  {^272^^^^^270^276^if  ( currentIterator == null )  {^[CLASS] IteratorChain  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Iterator  currentIterator  element  first  iterator  lastUsedIterator  second  boolean  isLocked  Queue  iteratorChain  
