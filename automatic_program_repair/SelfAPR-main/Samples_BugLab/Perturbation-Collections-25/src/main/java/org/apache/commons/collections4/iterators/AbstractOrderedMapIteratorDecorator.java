[BugLab_Wrong_Operator]^if  ( iterator != null )  {^43^^^^^41^47^if  ( iterator == null )  {^[CLASS] AbstractOrderedMapIteratorDecorator  [METHOD] <init> [RETURN_TYPE] OrderedMapIterator)   OrderedMapIterator<K, V> iterator [VARIABLES] OrderedMapIterator  iterator  boolean  
[BugLab_Variable_Misuse]^return this.getKey (  ) ;^87^^^^^86^88^return iterator.getKey (  ) ;^[CLASS] AbstractOrderedMapIteratorDecorator  [METHOD] getKey [RETURN_TYPE] K   [VARIABLES] OrderedMapIterator  iterator  boolean  
[BugLab_Argument_Swapping]^return obj.setValue ( iterator ) ;^97^^^^^96^98^return iterator.setValue ( obj ) ;^[CLASS] AbstractOrderedMapIteratorDecorator  [METHOD] setValue [RETURN_TYPE] V   final V obj [VARIABLES] OrderedMapIterator  iterator  V  obj  boolean  
