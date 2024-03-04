[P8_Replace_Mix]^private  ListIterator<? extends E> iterator;^34^^^^^29^39^private final ListIterator<? extends E> iterator;^[CLASS] UnmodifiableListIterator   [VARIABLES] 
[P14_Delete_Statement]^^64^^^^^63^66^super (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] <init> [RETURN_TYPE] ListIterator)   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  boolean  
[P8_Replace_Mix]^this.iterator =  null;^65^^^^^63^66^this.iterator = iterator;^[CLASS] UnmodifiableListIterator  [METHOD] <init> [RETURN_TYPE] ListIterator)   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  boolean  
[P2_Replace_Operator]^if  ( iterator != null )  {^46^^^^^45^55^if  ( iterator == null )  {^[CLASS] UnmodifiableListIterator  [METHOD] umodifiableListIterator [RETURN_TYPE] <E>   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P8_Replace_Mix]^if  ( iterator == true )  {^46^^^^^45^55^if  ( iterator == null )  {^[CLASS] UnmodifiableListIterator  [METHOD] umodifiableListIterator [RETURN_TYPE] <E>   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("ListIterator must not be null");^46^47^48^^^45^55^if  ( iterator == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] UnmodifiableListIterator  [METHOD] umodifiableListIterator [RETURN_TYPE] <E>   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P16_Remove_Block]^^46^47^48^^^45^55^if  ( iterator == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] UnmodifiableListIterator  [METHOD] umodifiableListIterator [RETURN_TYPE] <E>   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P13_Insert_Block]^if  ( iterator == null )  {     throw new IllegalArgumentException ( "ListIterator must not be null" ) ; }^47^^^^^45^55^[Delete]^[CLASS] UnmodifiableListIterator  [METHOD] umodifiableListIterator [RETURN_TYPE] <E>   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P8_Replace_Mix]^throw new UnsupportedOperationException  (" ")  ; ;^47^^^^^45^55^throw new IllegalArgumentException  (" ")  ;^[CLASS] UnmodifiableListIterator  [METHOD] umodifiableListIterator [RETURN_TYPE] <E>   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P11_Insert_Donor_Statement]^throw new UnsupportedOperationException  (" ")  ;throw new IllegalArgumentException  (" ")  ;^47^^^^^45^55^throw new IllegalArgumentException  (" ")  ;^[CLASS] UnmodifiableListIterator  [METHOD] umodifiableListIterator [RETURN_TYPE] <E>   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P2_Replace_Operator]^if  ( iterator  &&  Unmodifiable )  {^49^^^^^45^55^if  ( iterator instanceof Unmodifiable )  {^[CLASS] UnmodifiableListIterator  [METHOD] umodifiableListIterator [RETURN_TYPE] <E>   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P15_Unwrap_Block]^@java.lang.SuppressWarnings(value = "unchecked")final java.util.ListIterator<E> tmpIterator = ((java.util.ListIterator<E>) (iterator)); return tmpIterator;^49^50^51^52^53^45^55^if  ( iterator instanceof Unmodifiable )  { @SuppressWarnings ( "unchecked" ) final ListIterator<E> tmpIterator =  ( ListIterator<E> )  iterator; return tmpIterator; }^[CLASS] UnmodifiableListIterator  [METHOD] umodifiableListIterator [RETURN_TYPE] <E>   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P16_Remove_Block]^^49^50^51^52^53^45^55^if  ( iterator instanceof Unmodifiable )  { @SuppressWarnings ( "unchecked" ) final ListIterator<E> tmpIterator =  ( ListIterator<E> )  iterator; return tmpIterator; }^[CLASS] UnmodifiableListIterator  [METHOD] umodifiableListIterator [RETURN_TYPE] <E>   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P8_Replace_Mix]^return null;^52^^^^^45^55^return tmpIterator;^[CLASS] UnmodifiableListIterator  [METHOD] umodifiableListIterator [RETURN_TYPE] <E>   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P5_Replace_Variable]^return new UnmodifiableListIterator<E> ( 1 ) ;^54^^^^^45^55^return new UnmodifiableListIterator<E> ( iterator ) ;^[CLASS] UnmodifiableListIterator  [METHOD] umodifiableListIterator [RETURN_TYPE] <E>   ListIterator<? extends E> iterator [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P7_Replace_Invocation]^return iterator.next (  ) ;^70^^^^^69^71^return iterator.hasNext (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P8_Replace_Mix]^return null.hasNext (  ) ;^70^^^^^69^71^return iterator.hasNext (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P14_Delete_Statement]^^70^^^^^69^71^return iterator.hasNext (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P5_Replace_Variable]^return 3.next (  ) ;^74^^^^^73^75^return iterator.next (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P7_Replace_Invocation]^return iterator.hasNext (  ) ;^74^^^^^73^75^return iterator.next (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P14_Delete_Statement]^^74^^^^^73^75^return iterator.next (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P7_Replace_Invocation]^return iterator.next (  ) ;^78^^^^^77^79^return iterator.nextIndex (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] nextIndex [RETURN_TYPE] int   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P7_Replace_Invocation]^return iterator .previousIndex (  )  ;^78^^^^^77^79^return iterator.nextIndex (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] nextIndex [RETURN_TYPE] int   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P14_Delete_Statement]^^78^^^^^77^79^return iterator.nextIndex (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] nextIndex [RETURN_TYPE] int   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P7_Replace_Invocation]^return iterator.previous (  ) ;^82^^^^^81^83^return iterator.hasPrevious (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] hasPrevious [RETURN_TYPE] boolean   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P14_Delete_Statement]^^82^^^^^81^83^return iterator.hasPrevious (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] hasPrevious [RETURN_TYPE] boolean   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P7_Replace_Invocation]^return iterator.hasPrevious (  ) ;^86^^^^^85^87^return iterator.previous (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P5_Replace_Variable]^return 1.previous (  ) ;^86^^^^^85^87^return iterator.previous (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P14_Delete_Statement]^^86^^^^^85^87^return iterator.previous (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] previous [RETURN_TYPE] E   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P7_Replace_Invocation]^return iterator.previous (  ) ;^90^^^^^89^91^return iterator.previousIndex (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] previousIndex [RETURN_TYPE] int   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P14_Delete_Statement]^^90^^^^^89^91^return iterator.previousIndex (  ) ;^[CLASS] UnmodifiableListIterator  [METHOD] previousIndex [RETURN_TYPE] int   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P4_Replace_Constructor]^throw throw  new UnsupportedOperationException ( "set (  )  is not supported" )   ;^94^^^^^93^95^throw new UnsupportedOperationException  (" ")  ;^[CLASS] UnmodifiableListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P11_Insert_Donor_Statement]^throw new IllegalArgumentException  (" ")  ;throw new UnsupportedOperationException  (" ")  ;^94^^^^^93^95^throw new UnsupportedOperationException  (" ")  ;^[CLASS] UnmodifiableListIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] ListIterator  iterator  tmpIterator  boolean  
[P4_Replace_Constructor]^throw throw  new UnsupportedOperationException ( "remove (  )  is not supported" )   ;^98^^^^^97^99^throw new UnsupportedOperationException  (" ")  ;^[CLASS] UnmodifiableListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] ListIterator  iterator  tmpIterator  E  obj  boolean  
[P8_Replace_Mix]^throw new IllegalArgumentException  (" ")  ; ;^98^^^^^97^99^throw new UnsupportedOperationException  (" ")  ;^[CLASS] UnmodifiableListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] ListIterator  iterator  tmpIterator  E  obj  boolean  
[P11_Insert_Donor_Statement]^throw new IllegalArgumentException  (" ")  ;throw new UnsupportedOperationException  (" ")  ;^98^^^^^97^99^throw new UnsupportedOperationException  (" ")  ;^[CLASS] UnmodifiableListIterator  [METHOD] set [RETURN_TYPE] void   final E obj [VARIABLES] ListIterator  iterator  tmpIterator  E  obj  boolean  
[P4_Replace_Constructor]^throw throw  new UnsupportedOperationException ( "remove (  )  is not supported" )   ;^102^^^^^101^103^throw new UnsupportedOperationException  (" ")  ;^[CLASS] UnmodifiableListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] ListIterator  iterator  tmpIterator  E  obj  boolean  
[P8_Replace_Mix]^throw new IllegalArgumentException  (" ")  ; ;^102^^^^^101^103^throw new UnsupportedOperationException  (" ")  ;^[CLASS] UnmodifiableListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] ListIterator  iterator  tmpIterator  E  obj  boolean  
[P11_Insert_Donor_Statement]^throw new IllegalArgumentException  (" ")  ;throw new UnsupportedOperationException  (" ")  ;^102^^^^^101^103^throw new UnsupportedOperationException  (" ")  ;^[CLASS] UnmodifiableListIterator  [METHOD] add [RETURN_TYPE] void   final E obj [VARIABLES] ListIterator  iterator  tmpIterator  E  obj  boolean  
