[P3_Replace_Literal]^private boolean beforeFirst = false;^36^^^^^31^41^private boolean beforeFirst = true;^[CLASS] SingletonIterator   [VARIABLES] 
[P8_Replace_Mix]^private boolean beforeFirst  = null ;^36^^^^^31^41^private boolean beforeFirst = true;^[CLASS] SingletonIterator   [VARIABLES] 
[P3_Replace_Literal]^private boolean removed = true;^38^^^^^33^43^private boolean removed = false;^[CLASS] SingletonIterator   [VARIABLES] 
[P8_Replace_Mix]^private boolean removed ;^38^^^^^33^43^private boolean removed = false;^[CLASS] SingletonIterator   [VARIABLES] 
[P3_Replace_Literal]^this ( object, false ) ;^49^^^^^48^50^this ( object, true ) ;^[CLASS] SingletonIterator  [METHOD] <init> [RETURN_TYPE] SingletonIterator(E)   final E object [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P14_Delete_Statement]^^49^^^^^48^50^this ( object, true ) ;^[CLASS] SingletonIterator  [METHOD] <init> [RETURN_TYPE] SingletonIterator(E)   final E object [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P14_Delete_Statement]^^61^^^^^60^64^super (  ) ;^[CLASS] SingletonIterator  [METHOD] <init> [RETURN_TYPE] SingletonIterator(E,boolean)   final E object final boolean removeAllowed [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P8_Replace_Mix]^this.object =  null;^62^^^^^60^64^this.object = object;^[CLASS] SingletonIterator  [METHOD] <init> [RETURN_TYPE] SingletonIterator(E,boolean)   final E object final boolean removeAllowed [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P5_Replace_Variable]^this.removeAllowed = removed;^63^^^^^60^64^this.removeAllowed = removeAllowed;^[CLASS] SingletonIterator  [METHOD] <init> [RETURN_TYPE] SingletonIterator(E,boolean)   final E object final boolean removeAllowed [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P2_Replace_Operator]^return beforeFirst || !removed;^75^^^^^74^76^return beforeFirst && !removed;^[CLASS] SingletonIterator  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P5_Replace_Variable]^return removeAllowed && !removed;^75^^^^^74^76^return beforeFirst && !removed;^[CLASS] SingletonIterator  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P8_Replace_Mix]^returnremoved ;^75^^^^^74^76^return beforeFirst && !removed;^[CLASS] SingletonIterator  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P12_Insert_Condition]^if  ( !beforeFirst || removed )  { return beforeFirst && !removed; }^75^^^^^74^76^return beforeFirst && !removed;^[CLASS] SingletonIterator  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P2_Replace_Operator]^if  ( !beforeFirst && removed )  {^88^^^^^87^93^if  ( !beforeFirst || removed )  {^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P5_Replace_Variable]^if  ( !removeAllowed || removed )  {^88^^^^^87^93^if  ( !beforeFirst || removed )  {^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P5_Replace_Variable]^if  ( !beforeFirst || removeAllowed )  {^88^^^^^87^93^if  ( !beforeFirst || removed )  {^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P6_Replace_Expression]^if  ( !beforeFirst ) {^88^^^^^87^93^if  ( !beforeFirst || removed )  {^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P6_Replace_Expression]^if  (  removed )  {^88^^^^^87^93^if  ( !beforeFirst || removed )  {^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P8_Replace_Mix]^if  ( !removeAllowed ) {^88^^^^^87^93^if  ( !beforeFirst || removed )  {^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P15_Unwrap_Block]^throw new java.util.NoSuchElementException();^88^89^90^^^87^93^if  ( !beforeFirst || removed )  { throw new NoSuchElementException  (" ")  ; }^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P16_Remove_Block]^^88^89^90^^^87^93^if  ( !beforeFirst || removed )  { throw new NoSuchElementException  (" ")  ; }^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P13_Insert_Block]^if  (  ( removed )  ||  ( beforeFirst )  )  {     throw new IllegalStateException (  ) ; }^88^^^^^87^93^[Delete]^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P11_Insert_Donor_Statement]^throw new UnsupportedOperationException  (" ")  ;throw new NoSuchElementException  (" ")  ;^89^^^^^87^93^throw new NoSuchElementException  (" ")  ;^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P11_Insert_Donor_Statement]^throw new IllegalStateException  (" ")  ;throw new NoSuchElementException  (" ")  ;^89^^^^^87^93^throw new NoSuchElementException  (" ")  ;^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P3_Replace_Literal]^beforeFirst = true;^91^^^^^87^93^beforeFirst = false;^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P11_Insert_Donor_Statement]^beforeFirst = true;beforeFirst = false;^91^^^^^87^93^beforeFirst = false;^[CLASS] SingletonIterator  [METHOD] next [RETURN_TYPE] E   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P5_Replace_Variable]^if  ( removed )  {^105^^^^^104^114^if  ( removeAllowed )  {^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P15_Unwrap_Block]^if ((removed) || (beforeFirst)) {    throw new java.lang.IllegalStateException();}; object = null; removed = true;^105^106^107^108^^104^114^if  ( removeAllowed )  { if  ( removed || beforeFirst )  { throw new IllegalStateException  (" ")  ; }^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P16_Remove_Block]^^105^106^107^108^^104^114^if  ( removeAllowed )  { if  ( removed || beforeFirst )  { throw new IllegalStateException  (" ")  ; }^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P8_Replace_Mix]^return 0;^112^^^^^104^114^throw new UnsupportedOperationException  (" ")  ;^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P11_Insert_Donor_Statement]^throw new IllegalStateException  (" ")  ;throw new UnsupportedOperationException  (" ")  ;^112^^^^^104^114^throw new UnsupportedOperationException  (" ")  ;^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P11_Insert_Donor_Statement]^throw new NoSuchElementException  (" ")  ;throw new UnsupportedOperationException  (" ")  ;^112^^^^^104^114^throw new UnsupportedOperationException  (" ")  ;^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P2_Replace_Operator]^if  ( removed && beforeFirst )  {^106^^^^^104^114^if  ( removed || beforeFirst )  {^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P5_Replace_Variable]^if  ( removeAllowed || beforeFirst )  {^106^^^^^104^114^if  ( removed || beforeFirst )  {^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P5_Replace_Variable]^if  ( removed || removeAllowed )  {^106^^^^^104^114^if  ( removed || beforeFirst )  {^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P5_Replace_Variable]^if  ( beforeFirst || removed )  {^106^^^^^104^114^if  ( removed || beforeFirst )  {^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P6_Replace_Expression]^if  ( removed ) {^106^^^^^104^114^if  ( removed || beforeFirst )  {^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P6_Replace_Expression]^if  (  beforeFirst )  {^106^^^^^104^114^if  ( removed || beforeFirst )  {^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P8_Replace_Mix]^if  ( removeAllowed ) {^106^^^^^104^114^if  ( removed || beforeFirst )  {^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P15_Unwrap_Block]^throw new java.lang.IllegalStateException();^106^107^108^^^104^114^if  ( removed || beforeFirst )  { throw new IllegalStateException  (" ")  ; }^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P16_Remove_Block]^^106^107^108^^^104^114^if  ( removed || beforeFirst )  { throw new IllegalStateException  (" ")  ; }^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P13_Insert_Block]^if  (  ( ! ( beforeFirst )  )  ||  ( removed )  )  {     throw new NoSuchElementException (  ) ; }^106^^^^^104^114^[Delete]^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P11_Insert_Donor_Statement]^throw new UnsupportedOperationException  (" ")  ;throw new IllegalStateException  (" ")  ;^107^^^^^104^114^throw new IllegalStateException  (" ")  ;^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P11_Insert_Donor_Statement]^throw new NoSuchElementException  (" ")  ;throw new IllegalStateException  (" ")  ;^107^^^^^104^114^throw new IllegalStateException  (" ")  ;^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P13_Insert_Block]^if  (  ( removed )  ||  ( beforeFirst )  )  {     throw new IllegalStateException (  ) ; }^107^^^^^104^114^[Delete]^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P8_Replace_Mix]^object = false;^109^^^^^104^114^object = null;^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P3_Replace_Literal]^removed = false;^110^^^^^104^114^removed = true;^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P11_Insert_Donor_Statement]^beforeFirst = true;removed = true;^110^^^^^104^114^removed = true;^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P8_Replace_Mix]^throw new UnsupportedOperationException  (" ")  ; ;^107^^^^^104^114^throw new IllegalStateException  (" ")  ;^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P8_Replace_Mix]^object = this;^109^^^^^104^114^object = null;^[CLASS] SingletonIterator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P3_Replace_Literal]^beforeFirst = false;^120^^^^^119^121^beforeFirst = true;^[CLASS] SingletonIterator  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P11_Insert_Donor_Statement]^removed = true;beforeFirst = true;^120^^^^^119^121^beforeFirst = true;^[CLASS] SingletonIterator  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
[P11_Insert_Donor_Statement]^beforeFirst = false;beforeFirst = true;^120^^^^^119^121^beforeFirst = true;^[CLASS] SingletonIterator  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] E  object  boolean  beforeFirst  removeAllowed  removed  
