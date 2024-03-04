[P14_Delete_Statement]^^61^^^^^60^62^this ( new StringReader ( json )  ) ;^[CLASS] JsonStreamParser  [METHOD] <init> [RETURN_TYPE] String)   String json [VARIABLES] JsonReader  parser  Object  lock  String  json  boolean  
[P8_Replace_Mix]^parser =  new JsonReader ( null ) ;^69^^^^^68^72^parser = new JsonReader ( reader ) ;^[CLASS] JsonStreamParser  [METHOD] <init> [RETURN_TYPE] Reader)   Reader reader [VARIABLES] JsonReader  parser  Reader  reader  Object  lock  boolean  
[P3_Replace_Literal]^parser.setLenient ( false ) ;^70^^^^^68^72^parser.setLenient ( true ) ;^[CLASS] JsonStreamParser  [METHOD] <init> [RETURN_TYPE] Reader)   Reader reader [VARIABLES] JsonReader  parser  Reader  reader  Object  lock  boolean  
[P7_Replace_Invocation]^parser.JsonReader ( true ) ;^70^^^^^68^72^parser.setLenient ( true ) ;^[CLASS] JsonStreamParser  [METHOD] <init> [RETURN_TYPE] Reader)   Reader reader [VARIABLES] JsonReader  parser  Reader  reader  Object  lock  boolean  
[P8_Replace_Mix]^parser .JsonReader ( reader )  ;^70^^^^^68^72^parser.setLenient ( true ) ;^[CLASS] JsonStreamParser  [METHOD] <init> [RETURN_TYPE] Reader)   Reader reader [VARIABLES] JsonReader  parser  Reader  reader  Object  lock  boolean  
[P14_Delete_Statement]^^70^^^^^68^72^parser.setLenient ( true ) ;^[CLASS] JsonStreamParser  [METHOD] <init> [RETURN_TYPE] Reader)   Reader reader [VARIABLES] JsonReader  parser  Reader  reader  Object  lock  boolean  
[P8_Replace_Mix]^lock  =  lock ;^71^^^^^68^72^lock = new Object (  ) ;^[CLASS] JsonStreamParser  [METHOD] <init> [RETURN_TYPE] Reader)   Reader reader [VARIABLES] JsonReader  parser  Reader  reader  Object  lock  boolean  
[P15_Unwrap_Block]^throw new java.util.NoSuchElementException();^82^83^84^^^81^95^if  ( !hasNext (  )  )  { throw new NoSuchElementException  (" ")  ; }^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P16_Remove_Block]^^82^83^84^^^81^95^if  ( !hasNext (  )  )  { throw new NoSuchElementException  (" ")  ; }^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P13_Insert_Block]^if  ( ! ( hasNext (  )  )  )  {     throw new NoSuchElementException (  ) ; }^83^^^^^81^95^[Delete]^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P11_Insert_Donor_Statement]^throw new JsonSyntaxException  (" ")  ;throw new NoSuchElementException  (" ")  ;^83^^^^^81^95^throw new NoSuchElementException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P11_Insert_Donor_Statement]^throw new UnsupportedOperationException  (" ")  ;throw new NoSuchElementException  (" ")  ;^83^^^^^81^95^throw new NoSuchElementException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P11_Insert_Donor_Statement]^throw new JsonParseException  (" ")  ;throw new NoSuchElementException  (" ")  ;^83^^^^^81^95^throw new NoSuchElementException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P11_Insert_Donor_Statement]^throw new JsonIOException  (" ")  ;throw new NoSuchElementException  (" ")  ;^83^^^^^81^95^throw new NoSuchElementException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P14_Delete_Statement]^^87^^^^^81^95^return Streams.parse ( parser ) ;^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P8_Replace_Mix]^throw new JsonSyntaxException  (" ")  ; ;^91^^^^^81^95^throw new JsonParseException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P11_Insert_Donor_Statement]^throw new JsonSyntaxException  (" ")  ;throw new JsonParseException  (" ")  ;^91^^^^^81^95^throw new JsonParseException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P11_Insert_Donor_Statement]^throw new UnsupportedOperationException  (" ")  ;throw new JsonParseException  (" ")  ;^91^^^^^81^95^throw new JsonParseException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P11_Insert_Donor_Statement]^throw new NoSuchElementException  (" ")  ;throw new JsonParseException  (" ")  ;^91^^^^^81^95^throw new JsonParseException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P11_Insert_Donor_Statement]^throw new JsonIOException  (" ")  ;throw new JsonParseException  (" ")  ;^91^^^^^81^95^throw new JsonParseException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P8_Replace_Mix]^throw new JsonParseException  (" ")  ; ;^93^^^^^81^95^throw e.getCause  (" ")   : e;^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P14_Delete_Statement]^^93^^^^^81^95^throw e.getCause  (" ")   : e;^[CLASS] JsonStreamParser  [METHOD] next [RETURN_TYPE] JsonElement   [VARIABLES] JsonReader  parser  Object  lock  boolean  JsonParseException  e  StackOverflowError  e  OutOfMemoryError  e  
[P2_Replace_Operator]^return parser.peek (  )  == JsonToken.END_DOCUMENT;^105^^^^^102^112^return parser.peek (  )  != JsonToken.END_DOCUMENT;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P8_Replace_Mix]^return parser.peek (  )   ;^105^^^^^102^112^return parser.peek (  )  != JsonToken.END_DOCUMENT;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P14_Delete_Statement]^^105^^^^^102^112^return parser.peek (  )  != JsonToken.END_DOCUMENT;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P2_Replace_Operator]^return parser.peek (  )  <= JsonToken.END_DOCUMENT;^105^^^^^102^112^return parser.peek (  )  != JsonToken.END_DOCUMENT;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P8_Replace_Mix]^return parser .JsonReader ( 0 )    ;^105^^^^^102^112^return parser.peek (  )  != JsonToken.END_DOCUMENT;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P4_Replace_Constructor]^throw throw  new JsonSyntaxException ( e )   ;^109^^^^^102^112^throw new JsonIOException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P8_Replace_Mix]^throw new UnsupportedOperationException  (" ")  ; ;^107^^^^^102^112^throw new JsonSyntaxException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P11_Insert_Donor_Statement]^throw new UnsupportedOperationException  (" ")  ;throw new JsonSyntaxException  (" ")  ;^107^^^^^102^112^throw new JsonSyntaxException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P11_Insert_Donor_Statement]^throw new JsonParseException  (" ")  ;throw new JsonSyntaxException  (" ")  ;^107^^^^^102^112^throw new JsonSyntaxException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P11_Insert_Donor_Statement]^throw new NoSuchElementException  (" ")  ;throw new JsonSyntaxException  (" ")  ;^107^^^^^102^112^throw new JsonSyntaxException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P11_Insert_Donor_Statement]^throw new JsonIOException  (" ")  ;throw new JsonSyntaxException  (" ")  ;^107^^^^^102^112^throw new JsonSyntaxException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P8_Replace_Mix]^throw new JsonSyntaxException  (" ")  ; ;^109^^^^^102^112^throw new JsonIOException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P11_Insert_Donor_Statement]^throw new JsonSyntaxException  (" ")  ;throw new JsonIOException  (" ")  ;^109^^^^^102^112^throw new JsonIOException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P11_Insert_Donor_Statement]^throw new UnsupportedOperationException  (" ")  ;throw new JsonIOException  (" ")  ;^109^^^^^102^112^throw new JsonIOException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P11_Insert_Donor_Statement]^throw new JsonParseException  (" ")  ;throw new JsonIOException  (" ")  ;^109^^^^^102^112^throw new JsonIOException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P11_Insert_Donor_Statement]^throw new NoSuchElementException  (" ")  ;throw new JsonIOException  (" ")  ;^109^^^^^102^112^throw new JsonIOException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] JsonReader  parser  Object  lock  IOException  e  boolean  MalformedJsonException  e  
[P11_Insert_Donor_Statement]^throw new JsonSyntaxException  (" ")  ;throw new UnsupportedOperationException  (" ")  ;^120^^^^^119^121^throw new UnsupportedOperationException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] JsonReader  parser  Object  lock  boolean  
[P11_Insert_Donor_Statement]^throw new JsonParseException  (" ")  ;throw new UnsupportedOperationException  (" ")  ;^120^^^^^119^121^throw new UnsupportedOperationException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] JsonReader  parser  Object  lock  boolean  
[P11_Insert_Donor_Statement]^throw new NoSuchElementException  (" ")  ;throw new UnsupportedOperationException  (" ")  ;^120^^^^^119^121^throw new UnsupportedOperationException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] JsonReader  parser  Object  lock  boolean  
[P11_Insert_Donor_Statement]^throw new JsonIOException  (" ")  ;throw new UnsupportedOperationException  (" ")  ;^120^^^^^119^121^throw new UnsupportedOperationException  (" ")  ;^[CLASS] JsonStreamParser  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] JsonReader  parser  Object  lock  boolean  
