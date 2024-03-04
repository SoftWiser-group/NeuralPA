[BugLab_Variable_Misuse]^localData = copy ( localData ) ;^64^^^^^63^65^localData = copy ( data ) ;^[CLASS] UnrecognizedExtraField  [METHOD] setLocalFileDataData [RETURN_TYPE] void   byte[] data [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Argument_Swapping]^return new ZipShort ( localData ) ;^72^^^^^71^73^return new ZipShort ( localData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getLocalFileDataLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Argument_Swapping]^return new ZipShort ( localData.length.length ) ;^72^^^^^71^73^return new ZipShort ( localData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getLocalFileDataLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Variable_Misuse]^return new ZipShort ( data.length ) ;^72^^^^^71^73^return new ZipShort ( localData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getLocalFileDataLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Variable_Misuse]^return copy ( data ) ;^80^^^^^79^81^return copy ( localData ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getLocalFileDataData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Variable_Misuse]^centralData = copy ( localData ) ;^94^^^^^93^95^centralData = copy ( data ) ;^[CLASS] UnrecognizedExtraField  [METHOD] setCentralDirectoryData [RETURN_TYPE] void   byte[] data [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Variable_Misuse]^if  ( localData != null )  {^103^^^^^102^107^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Wrong_Operator]^if  ( centralData == null )  {^103^^^^^102^107^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Variable_Misuse]^return new ZipShort ( localData.length ) ;^104^^^^^102^107^return new ZipShort ( centralData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Argument_Swapping]^return new ZipShort ( centralData ) ;^104^^^^^102^107^return new ZipShort ( centralData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Argument_Swapping]^return new ZipShort ( centralData.length.length ) ;^104^^^^^102^107^return new ZipShort ( centralData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Variable_Misuse]^return new ZipShort ( 2 ) ;^104^^^^^102^107^return new ZipShort ( centralData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Variable_Misuse]^if  ( localData != null )  {^114^^^^^113^118^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Wrong_Operator]^if  ( centralData == null )  {^114^^^^^113^118^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Variable_Misuse]^return copy ( localData ) ;^115^^^^^113^118^return copy ( centralData ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[BugLab_Variable_Misuse]^System.arraycopy ( localData, offset, tmp, 0, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Variable_Misuse]^System.arraycopy ( data, length, tmp, 0, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Variable_Misuse]^System.arraycopy ( data, offset, localData, 0, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Argument_Swapping]^System.arraycopy ( length, offset, tmp, 0, data ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Argument_Swapping]^System.arraycopy ( data, length, tmp, 0, offset ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Argument_Swapping]^System.arraycopy ( data, offset, length, 0, tmp ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Wrong_Literal]^System.arraycopy ( data, offset, tmp, 1, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Variable_Misuse]^setLocalFileDataData ( localData ) ;^129^^^^^126^130^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Variable_Misuse]^System.arraycopy ( data, offset, localData, 0, length ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Argument_Swapping]^System.arraycopy ( tmp, offset, data, 0, length ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Argument_Swapping]^System.arraycopy ( data, length, tmp, 0, offset ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Argument_Swapping]^System.arraycopy ( data, tmp, offset, 0, length ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Argument_Swapping]^System.arraycopy ( length, offset, tmp, 0, data ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Variable_Misuse]^if  ( data == null )  {^143^^^^^138^146^if  ( localData == null )  {^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Wrong_Operator]^if  ( localData != null )  {^143^^^^^138^146^if  ( localData == null )  {^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Variable_Misuse]^setLocalFileDataData ( localData ) ;^144^^^^^138^146^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[BugLab_Variable_Misuse]^if  ( tmp != null )  {^149^^^^^148^155^if  ( from != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[BugLab_Wrong_Operator]^if  ( from == null )  {^149^^^^^148^155^if  ( from != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[BugLab_Variable_Misuse]^return tmp;^152^^^^^148^155^return to;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[BugLab_Variable_Misuse]^System.arraycopy ( to, 0, to, 0, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[BugLab_Variable_Misuse]^System.arraycopy ( from, 0, tmp, 0, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[BugLab_Argument_Swapping]^System.arraycopy ( to.length, 0, to, 0, from ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[BugLab_Argument_Swapping]^System.arraycopy ( to, 0, from, 0, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[BugLab_Argument_Swapping]^System.arraycopy ( from, 0, to.length, 0, to ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[BugLab_Wrong_Literal]^System.arraycopy ( from, -1, to, -1, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[BugLab_Wrong_Literal]^System.arraycopy ( from, 1, to, 1, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
